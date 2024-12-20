import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from torch.nn.functional import normalize
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

class SmoothCLIP:
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        noise_level: float = 0.1,
        num_samples: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.device = device

    def _add_noise(self, image_features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to normalized image features."""
        noise = torch.randn_like(image_features) * self.noise_level
        noisy_features = image_features + noise
        # Renormalize features as CLIP expects normalized vectors
        return normalize(noisy_features, dim=-1)

    def get_smooth_image_features(
        self, 
        image,
        return_confidence: bool = False
    ) -> Tuple[torch.Tensor, float]:
        """Get smoothed image features using randomized smoothing."""
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)['pixel_values']
        
        # Get base features
        with torch.no_grad():
            outputs = self.model(inputs)
            image_features = outputs.pooler_output
            image_features = normalize(image_features, dim=-1)

        # Apply randomized smoothing
        smoothed_features = torch.zeros_like(image_features)
        for _ in range(self.num_samples):
            noisy_features = self._add_noise(image_features)
            smoothed_features += noisy_features

        # Average and renormalize
        smoothed_features = normalize(smoothed_features / self.num_samples, dim=-1)
        
        if return_confidence:
            # Calculate confidence as cosine similarity between smoothed and original
            confidence = torch.cosine_similarity(smoothed_features, image_features, dim=-1)
            return smoothed_features, confidence.item()
        
        return smoothed_features

    def get_similarity(
        self, 
        image, 
        text_queries: List[str],
        return_confidence: bool = False
    ) -> Tuple[List[float], float]:
        """Get smoothed similarity scores between image and text queries."""
        # Get smoothed image features
        image_features, confidence = self.get_smooth_image_features(
            image, 
            return_confidence=True
        )

        # Process text queries
        text_inputs = self.processor(
            text=text_queries,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = normalize(text_features, dim=-1)

        # Calculate similarities
        similarities = torch.cosine_similarity(
            image_features.unsqueeze(1),
            text_features,
            dim=-1
        )

        if return_confidence:
            return similarities.cpu().tolist(), confidence
        
        return similarities.cpu().tolist()
    
class CLIPWithLinear(nn.Module):
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        linear_path: str = "linear-layers/layer_1.pth"
    ):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.linear = nn.Linear(512, 512)  # Assuming 512 is the CLIP embedding dimension
        
        # Load the linear layer weights
        state_dict = torch.load(linear_path)
        # Check if state dict has 'weight' and 'bias' directly or needs extraction
        if 'linear.weight' in state_dict:
            # Remove 'linear.' prefix from keys
            state_dict = {k.replace('linear.', ''): v for k, v in state_dict.items()}
        self.linear.load_state_dict(state_dict)
    
    def forward(self, **inputs):
        clip_outputs = self.clip(**inputs)
        linear_outputs = self.linear(clip_outputs.pooler_output)
        return linear_outputs

class SmoothCLIPWithLinear:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        linear_path: str = "linear-layers/layer_1.pth",
        noise_level: float = 0.1,
        num_samples: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = CLIPWithLinear(model_name, linear_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.device = device
        self.model.eval()

    def _add_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features."""
        noise = torch.randn_like(features) * self.noise_level
        return features + noise

    def get_smooth_features(
        self, 
        image, 
        return_confidence: bool = False
    ):
        """Get smoothed features using randomized smoothing."""
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)['pixel_values']
        
        # Get base features
        with torch.no_grad():
            outputs = self.model.clip(inputs)
            base_features = self.model.linear(outputs.pooler_output)

        # Apply randomized smoothing
        smoothed_features = torch.zeros_like(base_features)
        for _ in range(self.num_samples):
            # Add noise to CLIP features before linear layer
            with torch.no_grad():
                clip_outputs = self.model.clip(inputs)
                noisy_features = self._add_noise(clip_outputs.pooler_output)
                smoothed_features += self.model.linear(noisy_features)

        # Average the features
        smoothed_features = smoothed_features / self.num_samples
        
        if return_confidence:
            # Calculate confidence as cosine similarity between smoothed and original
            confidence = F.cosine_similarity(smoothed_features, base_features, dim=-1)
            return smoothed_features, confidence.item()
        
        return smoothed_features, None

    def process_image_directory(
        self, 
        image_dir: str = "images/"
    ) -> Tuple[torch.Tensor, List[str]]:
        """Process all images in a directory and return their smoothed features."""
        image_features = []
        image_paths = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path)
                
                # Get smoothed features
                features, _ = self.get_smooth_features(image)
                image_features.append(features)
                image_paths.append(image_path)
        
        if image_features:
            image_features = torch.cat(image_features, dim=0)
            
        return image_features, image_paths

    def detect_watermark(
        self,
        image_path: str,
        keys_dir: str = "keys/",
        confidence_threshold: float = 0.8
    ) -> Tuple[bool, float]:
        """
        Detect if image contains a watermark by comparing with key files.
        Returns (is_watermarked, confidence_score)
        """
        # Load and process target image
        image = Image.open(image_path)
        image_features, _ = self.get_smooth_features(image)
        
        # Process all key files
        key_features = []
        for key_file in os.listdir(keys_dir):
            if key_file.lower().endswith(('.pth', '.pt')):
                key_path = os.path.join(keys_dir, key_file)
                key = torch.load(key_path).to(self.device)
                key_features.append(key)
        
        if not key_features:
            raise ValueError("No key files found in keys directory")
            
        key_features = torch.stack(key_features)
        
        # Calculate similarities with all keys
        similarities = F.cosine_similarity(
            image_features.unsqueeze(0),  # [1, dim]
            key_features,                 # [num_keys, dim]
            dim=1
        )
        
        # Check if maximum similarity exceeds threshold
        max_similarity = similarities.max().item()
        is_watermarked = max_similarity >= confidence_threshold
        
        return is_watermarked, max_similarity

# Example usage
def main():
    model = SmoothCLIPWithLinear(
        noise_level=0.1,
        num_samples=100
    )
    
    # Process single image
    image_path = "path/to/image.jpg"
    is_watermarked, confidence = model.detect_watermark(
        image_path,
        confidence_threshold=0.8
    )
    print(f"Image {image_path}:")
    print(f"Watermarked: {is_watermarked}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()