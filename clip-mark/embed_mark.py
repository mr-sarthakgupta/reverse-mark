import torch
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import os
from autoattack import AutoAttack
import torch.nn.functional as F
import torchvision.transforms as T

def process_images(image_dir="images/"):
    # Initialize CLIP vision model and processor
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    image_embeddings = []
    image_paths = []
    
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            
            # Preprocess image using CLIP processor
            inputs = processor(images=image, return_tensors="pt")
            inputs = inputs.to(device)['pixel_values']
            
            # Get image embeddings
            with torch.no_grad():
                outputs  = model(inputs)
                image_features = outputs.pooler_output
            
            image_embeddings.append(image_features)
            image_paths.append(image_path)
            
    # Stack all embeddings into a single tensor
    if image_embeddings:
        image_embeddings = torch.cat(image_embeddings, dim=0)
        
    return image_embeddings, image_paths

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, 2)
        # Load the pre-trained weights
        weights = torch.load('linear-layers/layer_1.pth')
        self.projection.load_state_dict(weights)
    
    def forward(self, x):
        return self.projection(x)

class ProjectedCLIPAttacker(nn.Module):
    def __init__(self, clip_model, clip_processor, projector, target_points, eps=8/255, save_dir="embedded_images"):
        super().__init__()
        self.model = clip_model
        self.processor = clip_processor
        self.projector = projector
        self.target_points = target_points  # Shape: [N, 2] - the 2D target points
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize AutoAttack with custom loss
        self.adversary = AutoAttack(
            self,
            norm='Linf',
            eps=eps,
            version='custom',
            attacks_to_run=['apgd-ce']
        )
        
        # Transform to convert tensors back to PIL images
        self.to_pil = T.ToPILImage()
    
    def forward(self, x):
        # Process through CLIP and projector
        clip_outputs = self.model(x)
        projected = self.projector(clip_outputs.pooler_output)
        return projected
    
    def get_logits(self, x):
        # AutoAttack expects logits, so we compute negative L2 distances
        # More negative means further, more positive means closer
        projected = self.forward(x)
        distances = -torch.cdist(projected, self.target_points)
        return distances
    
    def attack_and_save(self, images, target_indices, original_filenames=None):
        """
        Attack images and save the results
        
        Args:
            images: List of PIL images
            target_indices: List of indices into self.target_points for each image
            original_filenames: Optional list of original filenames to use as base names
        """
        # Get adversarial images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to("cuda:0" if torch.cuda.is_available() else 'cpu')['pixel_values']
        
        # Create target labels
        targets = torch.tensor(target_indices).to("cuda:0" if torch.cuda.is_available() else 'cpu')
        
        # Run attack
        
        adv_images = self.adversary.run_standard_evaluation(
            inputs,
            targets,
            bs=len(images)
        )
        
        # If no filenames provided, generate generic ones
        if original_filenames is None:
            original_filenames = [f"image_{i}.png" for i in range(len(images))]
        
        # Save each adversarial image
        saved_paths = []
        for i, (adv_image, orig_name) in enumerate(zip(adv_images, original_filenames)):
            # Create filename with target index
            base_name = os.path.splitext(orig_name)[0]
            save_name = f"{base_name}_projected_target_{target_indices[i]}.png"
            save_path = os.path.join(self.save_dir, save_name)
            
            # Convert tensor to PIL image and save
            pil_image = self.to_pil(adv_image.squeeze(0).cpu())
            pil_image.save(save_path)
            saved_paths.append(save_path)
            
        return adv_images, saved_paths

class SoftmaxCLIP(nn.Module):
    def __init__(self, clip_model):
        super(SoftmaxCLIP, self).__init__()
        self.model = clip_model
        
    def forward(self, x):
        # Process through CLIP only
        clip_outputs = self.model(x)
        # return F.softmax(clip_outputs.pooler_output)
        return clip_outputs.pooler_output

class CLIPAttacker(nn.Module):
    def __init__(self, clip_model, clip_processor, target_embeddings, eps=8/255, save_dir="clip_images"):
        super().__init__()
        self.model = SoftmaxCLIP(clip_model)
        self.processor = clip_processor
        self.target_embeddings = target_embeddings  # Shape: [N, 768] - the CLIP embeddings
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize AutoAttack with custom loss
        self.adversary = AutoAttack(
            self,
            norm='Linf',
            eps=eps,
            version='standard'
        )
        
        # Transform to convert tensors back to PIL images
        self.to_pil = T.ToPILImage()
    
    def forward(self, x):
        # Process through CLIP only
        clip_outputs = self.model(x)
        # return F.softmax(clip_outputs)
        return clip_outputs
    
    def attack_and_save(self, images, target_indices, original_filenames=None):
        """
        Attack images and save the results
        
        Args:
            images: List of PIL images
            target_indices: List of indices into self.target_embeddings for each image
            original_filenames: Optional list of original filenames to use as base names
        """
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to("cuda:0" if torch.cuda.is_available() else 'cpu')['pixel_values'][0].unsqueeze(0)
        
        targets = target_indices.to("cuda:0" if torch.cuda.is_available() else 'cpu')
        
        logits = self.model(inputs)

        top_indices = torch.topk(logits, 25, dim=1).indices

        print("Top 25 highest value indices for each image:", top_indices)
        count = sum(1 for idx in target_indices if idx in top_indices)
        print(f"initial images : {count}")
                
        # Run attack
        adv_images = self.adversary.run_standard_evaluation(
            inputs,
            [625],
            bs=len(images)
        )   
        logits = self.model(adv_images)
        print("Top 25 highest value indices for each image after attack:", torch.topk(logits, 25, dim=1).indices)
        count = sum(1 for idx in target_indices if idx in top_indices)
        print(f"adv images : {count}")
        # print(adv_images.shape, inputs.shape)
        print(torch.sum(torch.abs(adv_images - inputs)))
        exit()

        # Print the top 25 indices for each adversarial image
        for i, logits in enumerate(logits):
            top_indices = torch.topk(logits, 25, dim=0).indices
            # print(f"Top 25 indices for adversarial image {i}: {top_indices}")
            count = sum(1 for idx in target_indices if idx in top_indices)
            print(f"{i}: {count}")
        
        # If no filenames provided, generate generic ones
        if original_filenames is None:
            original_filenames = [f"image_{i}.png" for i in range(len(images))]
        
        # Save each adversarial image
        saved_paths = []
        for i, (adv_image, orig_name) in enumerate(zip(adv_images, original_filenames)):
            # Create filename with target index
            base_name = os.path.splitext(orig_name)[0]
            save_name = f"{base_name}_clip_target_{target_indices[i]}.png"
            save_path = os.path.join(self.save_dir, save_name)
            
            # Convert tensor to PIL image and save
            pil_image = self.to_pil(adv_image.squeeze(0).cpu())
            pil_image.save(save_path)
            saved_paths.append(save_path)
            
        return adv_images, saved_paths

# Updated main block to demonstrate both attackers
if __name__ == "__main__":
    # Get original embeddings
    embeddings, paths = process_images()
    
    # Initialize models
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)    

    target_indices = torch.randint(0, int(embeddings.shape[-1]) - 1, (25, )).to(device)

    # Save target points
    os.makedirs('keys', exist_ok=True)
    torch.save(target_indices, 'keys/target_points.pt')

    clip_attacker = CLIPAttacker(
        clip_model=clip_model,
        clip_processor=processor,
        target_embeddings=embeddings,
        save_dir="embedded_images_clip"
    )
    
    # Attack using both methods
    original_images = [Image.open(path) for path in paths]
    original_filenames = [os.path.basename(path) for path in paths]
    
    # Attack and save with CLIP attacker
    adv_images_clip, saved_paths_clip = clip_attacker.attack_and_save(
        original_images,
        target_indices,
        original_filenames
    )
    
    print(f"Saved {len(saved_paths_clip)} CLIP adversarial images to {clip_attacker.save_dir}/")