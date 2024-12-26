import clip
from random import shuffle
import torch
from torchvision.transforms.functional import pil_to_tensor
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import os
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

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.projection = nn.Linear(input_dim, 2)
        # Load the pre-trained weights
        weights = torch.load('linear-layers/layer_1.pth')
        self.projection.load_state_dict(weights)
    
    def forward(self, x):
        return self.projection(x)

class CLIPFwd(nn.Module):
    def __init__(self):
        super(CLIPFwd, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda:0")
        
    def forward(self, x):

        print(x.device)
        print(self.preprocess(x).device)
        exit()

        return self.model.encode_image()

class CLIPAttacker(nn.Module):
    def __init__(self, target_embeddings, eps=8/255, save_dir="clip_images"):
        super().__init__()
        self.model = CLIPFwd()
        self.target_embeddings = target_embeddings  # Shape: [N, 768] - the CLIP embeddings
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Transform to convert tensors back to PIL images
        self.to_pil = T.ToPILImage()
<<<<<<< HEAD
=======
        self.adversary = AutoAttack(
            model=self.model,
            norm='Linf',
            eps=8/255,
            version='custom',
            # version='standard',
            device="cuda:0",
            attacks_to_run=['apgd-ce'],
            verbose=True, 
            
        )
>>>>>>> 344ede2997451873a8985736f09bdc976ede220b

    def loss_fn(self, outputs, target_indices):
        softmax_outputs = F.softmax(outputs, dim=-1)
        labels = torch.zeros_like(outputs)
        labels[:, target_indices] = 1
        return torch.norm(softmax_outputs - labels, p=1, dim=-1)
    
    def attack_and_save(self, images, target_indices, original_filenames=None):
        self.model.to("cpu")
        eps = 8/255
        alpha = 2/255

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        for image in images:
            # image = pil_to_tensor(image).unsqueeze(0).to("cuda:0").float()
            image = pil_to_tensor(image).unsqueeze(0).float().to("cuda:0") / 255

            # image = self.processor(images=image, return_tensors="pt", do_rescale = False)['pixel_values']
            adv_image = image.clone().detach()
            adv_image = adv_image + torch.empty_like(adv_image).uniform_(-eps, eps)
            adv_image = torch.clamp(adv_image, min = 0, max = 1).detach().to("cpu")
            for _ in range(100):  
                adv_image = adv_image.to("cpu")
                adv_image.requires_grad = True
                outputs = self.model(adv_image)
                loss = self.loss_fn(outputs, target_indices)              
                # Backward pass
                grad = torch.autograd.grad(loss, adv_image, retain_graph=False)[0]
                adv_image = adv_image.detach() + alpha * torch.sign(grad)
                delta = torch.clamp(adv_image.cuda() - image.cuda(), min=-eps, max=eps)
                adv_image = torch.clamp(image.cuda() + delta.cuda(), min=0, max=1).detach()
            
            print(torch.norm(adv_image.cuda() - image.cuda(), p=float('inf')))
            exit()
            
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
        target_embeddings=embeddings,
        save_dir="embedded_images_clip"
    )
    
    # Attack using both methods
    original_images = [Image.open(path) for path in paths]
    original_filenames = [os.path.basename(path) for path in paths]
    shuffle(original_images)
    # Attack and save with CLIP attacker
    adv_images_clip, saved_paths_clip = clip_attacker.attack_and_save(
        original_images[:1],
        target_indices,
        original_filenames
    )
    
    print(f"Saved {len(saved_paths_clip)} CLIP adversarial images to {clip_attacker.save_dir}/")