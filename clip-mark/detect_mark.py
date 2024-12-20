import torch
from transformers import CLIPProcessor, CLIPVisionModel
from torch.nn.functional import normalize
import numpy as np
from typing import List, Tuple

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
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get base features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
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
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Get smoothed features using randomized smoothing."""
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get base features
        with torch.no_grad():
            base_features = self.model(**inputs)

        # Apply randomized smoothing
        smoothed_features = torch.zeros_like(base_features)
        for _ in range(self.num_samples):
            # Add noise to CLIP features before linear layer
            with torch.no_grad():
                clip_outputs = self.model.clip(**inputs)
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


# Example usage
"""
smooth_clip = SmoothCLIP(
    noise_level=0.1,
    num_samples=100
)

# Load your image
image = Image.open("path_to_image.jpg")

# Get similarities with confidence
text_queries = ["a photo of a dog", "a photo of a cat"]
similarities, confidence = smooth_clip.get_similarity(
    image,
    text_queries,
    return_confidence=True
)

print(f"Similarities: {similarities}")
print(f"Confidence: {confidence}")
"""