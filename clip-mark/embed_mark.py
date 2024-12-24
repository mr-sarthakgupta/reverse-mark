from copy import deepcopy
from random import shuffle
import torch
from torchvision.transforms.functional import pil_to_tensor
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

    pathlist = list(os.listdir(image_dir))
    shuffle(pathlist)

    for filename in pathlist:
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

# class EmbeddingProjector(nn.Module):
#     def __init__(self, input_dim=512):
#         super().__init__()
#         self.projection = nn.Linear(input_dim, 2)
#         # Load the pre-trained weights
#         weights = torch.load('linear-layers/layer_1.pth')
#         self.projection.load_state_dict(weights)
    
#     def forward(self, x):
#         return self.projection(x)

# class ProjectedCLIPAttacker(nn.Module):
#     def __init__(self, clip_model, clip_processor, projector, target_points, eps=8/255, save_dir="embedded_images"):
#         super().__init__()
#         self.model = clip_model
#         self.processor = clip_processor
#         self.projector = projector
#         self.target_points = target_points  # Shape: [N, 2] - the 2D target points
#         self.save_dir = save_dir
        
#         # Create save directory if it doesn't exist
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Initialize AutoAttack with custom loss
#         self.adversary = AutoAttack(
#             self,
#             norm='Linf',
#             eps=eps,
#             version='custom',
#             attacks_to_run=['apgd-ce']
#         )
        
#         # Transform to convert tensors back to PIL images
#         self.to_pil = T.ToPILImage()
    
#     def forward(self, x):
#         # Process through CLIP and projector
#         clip_outputs = self.model(x)
#         projected = self.projector(clip_outputs.pooler_output)
#         return projected
    
#     def get_logits(self, x):
#         # AutoAttack expects logits, so we compute negative L2 distances
#         # More negative means further, more positive means closer
#         projected = self.forward(x)
#         distances = -torch.cdist(projected, self.target_points)
#         return distances
    
#     def attack_and_save(self, images, target_indices, original_filenames=None):
#         """
#         Attack images and save the results
        
#         Args:
#             images: List of PIL images
#             target_indices: List of indices into self.target_points for each image
#             original_filenames: Optional list of original filenames to use as base names
#         """
#         # Get adversarial images
#         inputs = self.processor(images=images, return_tensors="pt")
#         inputs = inputs.to("cuda:0" if torch.cuda.is_available() else 'cpu')['pixel_values']
        
#         # Create target labels
#         targets = torch.tensor(target_indices).to("cuda:0" if torch.cuda.is_available() else 'cpu')
        
#         # Run attack
        
#         adv_images = self.adversary.run_standard_evaluation(
#             inputs,
#             targets,
#             bs=len(images)
#         )
        
#         # If no filenames provided, generate generic ones
#         if original_filenames is None:
#             original_filenames = [f"image_{i}.png" for i in range(len(images))]
        
#         # Save each adversarial image
#         saved_paths = []
#         for i, (adv_image, orig_name) in enumerate(zip(adv_images, original_filenames)):
#             # Create filename with target index
#             base_name = os.path.splitext(orig_name)[0]
#             save_name = f"{base_name}_projected_target_{target_indices[i]}.png"
#             save_path = os.path.join(self.save_dir, save_name)
            
#             # Convert tensor to PIL image and save
#             pil_image = self.to_pil(adv_image.squeeze(0).cpu())
#             pil_image.save(save_path)
#             saved_paths.append(save_path)
            
#         return adv_images, saved_paths

class SoftmaxCLIP(nn.Module):
    def __init__(self, clip_model, processor):
        super(SoftmaxCLIP, self).__init__()
        self.model = clip_model
        self.processor = processor
        
    def forward(self, x):
        x = self.processor(images=x, return_tensors="pt")['pixel_values'].to("cuda:0" if torch.cuda.is_available() else 'cpu')
        clip_outputs = self.model(x)
        return clip_outputs.pooler_output

class CLIPAttacker(nn.Module):
    def __init__(self, clip_model, clip_processor, target_embeddings, eps=8/255, save_dir="clip_images"):
        super().__init__()
        self.model = SoftmaxCLIP(clip_model, clip_processor)
        self.target_embeddings = target_embeddings  # Shape: [N, 768] - the CLIP embeddings
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Transform to convert tensors back to PIL images
        self.to_pil = T.ToPILImage()
        self.adversary = AutoAttack(
            model=self.model,
            norm='Linf',
            eps=8/255,
            version='custom',
            # version='standard',
            device="cuda:0",
            attacks_to_run=['apgd-ce']
        )

    def loss_fn(self, outputs, target_indices):
        softmax_outputs = F.softmax(outputs, dim=-1)
        labels = torch.zeros_like(outputs)
        labels[:, target_indices] = 1
        return torch.norm(softmax_outputs - labels, p=1, dim=-1)

    def attack_and_save(self, images, target_indices, original_filenames=None): 
        images = pil_to_tensor(images).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else 'cpu').float()
        targets = target_indices.to("cuda:0" if torch.cuda.is_available() else 'cpu').squeeze()
        logits = self.model(images)
        top_indices = torch.topk(logits, 25, dim=1).indices
        print("Top 25 highest value indices for each image:", top_indices)
        count = sum(1 for idx in target_indices if idx in top_indices)
        print(f"initial images : {count}")
        
        print("targets: ", targets)
        
        # Run attack
        adv_images = self.adversary.run_standard_evaluation(
            images,
            # logits.squeeze(),
            targets,
            bs=len(images)
        )   
        logits = self.model(adv_images)
        print("Top 25 highest value indices for each image after attack:", torch.topk(logits, 25, dim=1).indices)
        count = sum(1 for idx in target_indices if idx in top_indices)
        print(f"adv images : {count}")
        print(torch.sum(torch.abs(adv_images - images)))

        exit()
        return adv_images, saved_paths

    
    # def attack_and_save(self, images, target_indices, original_filenames=None):

    #     for image in images:
    #         image = pil_to_tensor(image).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else 'cpu').float()
    #         og_image = image.clone()
    #         image.requires_grad = True
    #         optimizer = torch.optim.AdamW([image], lr=0.01)
    #         for _ in range(100):  # Number of attack iterations
    #             optimizer.zero_grad()
    #             outputs = self.model(self.processor(images=image, return_tensors="pt")['pixel_values'].to("cuda:0" if torch.cuda.is_available() else 'cpu'))
    #             loss = self.loss_fn(outputs, target_indices)              
    #             loss.backward()                
    #             optimizer.step()                
    #             image.data = torch.clamp(image.data, 0, 1)
                
    #         adv_images = image.detach()
    #         # saved_paths = []
    #         # for i, (adv_image, orig_name) in enumerate(zip(adv_images, original_filenames)):
    #         #     # Create filename with target index
    #         #     base_name = os.path.splitext(orig_name)[0]
    #         #     save_name = f"{base_name}_adv_target_{target_indices[i]}.png"
    #         #     save_path = os.path.join(self.save_dir, save_name)
                
    #         #     # Convert tensor to PIL image and save
    #         #     pil_image = self.to_pil(adv_image.squeeze(0).cpu())
    #         #     pil_image.save(save_path)
    #         #     saved_paths.append(save_path)

    #         current_outputs = self.model(self.processor(images=adv_images, return_tensors="pt")['pixel_values'].to("cuda:0" if torch.cuda.is_available() else 'cpu'))


    #         _, og_topk_indices = torch.topk(self.model(self.processor(images=og_image, return_tensors="pt")['pixel_values'].to("cuda:0" if torch.cuda.is_available() else 'cpu')).squeeze(), 25)
    #         _, adv_topk_indices = torch.topk(current_outputs.squeeze(), 25)

    #         og_in_target = sum([1 for idx in target_indices if idx in og_topk_indices])
    #         adv_in_target = sum([1 for idx in target_indices if idx in adv_topk_indices])
    #         print(f"Number of target indices in original top-k: {og_in_target}")
    #         print(f"Number of target indices in adversarial top-k: {adv_in_target}")
    #         print(f"original outs: {og_topk_indices}")
    #         print(f"adversarial outs: {adv_topk_indices}")
    #         print(f"target indices: {target_indices}")
    #         print(f"linf change in image: {torch.norm(og_image - adv_images, p = float('inf'))}")
    #         exit()

    #     return adv_images, saved_paths

if __name__ == "__main__":
    embeddings, paths = process_images()
    
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)   

    model = SoftmaxCLIP(clip_model, processor) 

    model.eval()

    clip_attacker = CLIPAttacker(
        clip_model=clip_model,
        clip_processor=processor,
        target_embeddings=embeddings,
        save_dir="embedded_images_clip"
    )
    
    original_images = [Image.open(path) for path in paths]
    original_filenames = [os.path.basename(path) for path in paths]
    # shuffle(original_images)

    im = deepcopy(original_images[1])

    im = pil_to_tensor(im).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else 'cpu').float()

    outs = model(im)

    target_indices = torch.topk(outs.squeeze(), 10).indices
    target_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0" if torch.cuda.is_available() else 'cpu')

    os.makedirs('keys', exist_ok=True)
    torch.save(target_indices, 'keys/target_points.pt')

    adv_images_clip, saved_paths_clip = clip_attacker.attack_and_save(
        original_images[0],
        target_indices,
        original_filenames
    )
    
    print(f"Saved {len(saved_paths_clip)} CLIP adversarial images to {clip_attacker.save_dir}/")