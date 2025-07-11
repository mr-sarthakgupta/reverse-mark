from open_clip.src import open_clip as clip
from random import shuffle
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import numpy as np
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import os
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/scratch/vidit_a_mfs.iitr/reverse-mark/clip-mark/open_clip/src')

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
        self.model, _, self.preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        
    def forward(self, x):
        return self.model.encode_image(self.preprocess(x))

class CLIPAttacker(nn.Module):
    def __init__(self, eps=8/255, save_dir="clip_images"):
        super().__init__()
        self.model = CLIPFwd()
        self.save_dir = save_dir
        self.softmax = nn.Softmax(dim=-1)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Transform to convert tensors back to PIL images
        self.to_pil = T.ToPILImage()

    def loss_fn(self, outputs, target_indices):
        softmax_outputs = self.softmax(outputs)
        dim1, dim2 = softmax_outputs.shape
        labels = torch.zeros(dim1, dim2)
        labels[:, target_indices] = 1
        if torch.isnan(softmax_outputs).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in the tensors")
        # return -1 * torch.norm(softmax_outputs - labels.cuda(), p=float('inf'), dim=-1)
        return -1 * torch.norm(softmax_outputs - labels.cuda(), p=2, dim=-1)
    
    def attack_and_save(self, images, target_indices, original_filenames=None):
        self.model.to("cuda:0")
        eps = 32/255
        alpha = 8/255

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        save_dirs = []
        
        num_fail = 0

        for i, image in enumerate(images):
            image = pil_to_tensor(image).unsqueeze(0).float().to("cuda:0") / 255
            
            if image.shape[1] != 3:
                num_fail += 1
                continue

            with torch.no_grad():
                og_out_maxes = torch.topk(self.model(image), dim=-1, k = 100).indices
            
            # image = self.processor(images=image, return_tensors="pt", do_rescale = False)['pixel_values']
            adv_images = image.clone().detach()
            adv_images = adv_images.to("cuda:0")
            adv_images.requires_grad = True

            all_transforms = []
            for _ in range(128):  
                transform = torch.nn.Sequential(
                            transforms.RandomResizedCrop(size = (adv_images.shape[-2], adv_images.shape[-1]), scale = (0.25, 1), ratio = (0.99, 1)),
                        )
                all_transforms.append(transform.to("cuda:0"))
            
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min = 0, max = 1).detach().to("cuda:0")
                                                                                        
            for _ in range(1024):
                adv_images.requires_grad = True
                for transform in all_transforms:
                    adv_images = torch.cat((adv_images, transform(adv_images[0].unsqueeze(0))), dim=0)
                outputs = self.model(adv_images)
                loss = self.loss_fn(outputs, target_indices).sum()
                # Backward pass
                grad = torch.autograd.grad(loss, adv_images, retain_graph=False)[0]
                adv_images = adv_images.detach() + alpha * torch.sign(grad)
                delta = torch.clamp(adv_images[0].cuda() - image.cuda(), min=-eps, max=eps)
                adv_images = torch.clamp(image.cuda() + delta.cuda(), min=0, max=1).detach()

                
            all_adv_images = adv_images.detach().to("cuda:0")
            all_adv_images.requires_grad = False
            all_og_images = image.detach().to("cuda:0")

            with torch.no_grad():
                og_outs = self.model(all_og_images)
                adv_outs = self.model(all_adv_images)
            og_count_single = sum([1 for idx in target_indices if idx in torch.topk(og_outs, dim=-1, k = 100).indices])
            adv_count_single = sum([1 for idx in target_indices if idx in torch.topk(adv_outs, dim=-1, k = 100).indices])

            print(f'with the single image: {i} | og count: {og_count_single} | adv count: {adv_count_single}')

            for transform in all_transforms:
                all_og_images = torch.cat((all_og_images, transform(image[0].unsqueeze(0))), dim=0)
                all_adv_images = torch.cat((all_adv_images, transform(adv_images[0].unsqueeze(0))), dim=0)

            og_count_sum = 0
            adv_count_sum = 0

            with torch.no_grad():
                for adv_img, ogd_img in zip(all_adv_images, all_og_images):
                    og_out_maxes = torch.topk(self.model(og_img.unsqueeze(0)), dim=-1, k = 100).indices
                    og_count = sum([1 for idx in target_indices if idx in og_out_maxes])
                    og_count_sum += og_count
                    adv_out_maxes = torch.topk(self.model(adv_img.unsqueeze(0)), dim=-1, k = 100).indices
                    adv_count = sum([1 for idx in target_indices if idx in adv_out_maxes])
                    adv_count_sum += adv_count
            
            # Count the number of target indices in both the original and adversarial outputs
            og_count = og_count_sum / len(all_og_images)
            adv_count = adv_count_sum / len(all_adv_images)

            print(f'with the og transforms: {i} | og count: {og_count} | adv count: {adv_count}')

            all_transforms = []
            for _ in range(128):  
                transform = torch.nn.Sequential(
                            transforms.RandomResizedCrop(size = (adv_images.shape[-2], adv_images.shape[-1]), scale = (0.25, 1), ratio = (0.99, 1)),
                        )
                all_transforms.append(transform.to("cuda:0"))
            

            for transform in all_transforms:
                all_og_images = torch.cat((all_og_images, transform(image[0].unsqueeze(0))), dim=0)
                all_adv_images = torch.cat((all_adv_images, transform(adv_images[0].unsqueeze(0))), dim=0)

            og_count_sum = 0
            adv_count_sum = 0

            with torch.no_grad():
                for adv_img, og_img in zip(all_adv_images, all_og_images):
                    og_out_maxes = torch.topk(self.model(og_img.unsqueeze(0)), dim=-1, k = 100).indices
                    og_count = sum([1 for idx in target_indices if idx in og_out_maxes])
                    og_count_sum += og_count
                    adv_out_maxes = torch.topk(self.model(adv_img.unsqueeze(0)), dim=-1, k = 100).indices
                    adv_count = sum([1 for idx in target_indices if idx in adv_out_maxes])
                    adv_count_sum += adv_count
            
            # Count the number of target indices in both the original and adversarial outputs
            og_count = og_count_sum / len(all_og_images)
            adv_count = adv_count_sum / len(all_adv_images)

            print(f'with the new transforms: {i} | og count: {og_count} | adv count: {adv_count}')

            
            
            # Create subdirectory for each image
            image_name = original_filenames[i] if original_filenames else f"image_{i}"
            image_dir = os.path.join(self.save_dir, image_name)
            os.makedirs(image_dir, exist_ok=True)
            
            # Save original image
            original_image_path = os.path.join(image_dir, f"{image_name}_original.png")
            self.to_pil(image.squeeze(0).cpu()).save(original_image_path)
            
            # Save adversarial image
            adv_image_path = os.path.join(image_dir, f"{image_name}_adversarial.png")
            self.to_pil(adv_images.squeeze(0).cpu()).save(adv_image_path)
            
            # Save counts to JSON file
            counts = {
                "original_count": og_count,
                "adversarial_count": adv_count
            }
            # print(i, f"Original count: {og_count}, Adversarial count: {adv_count}")
            json_path = os.path.join(image_dir, f"{image_name}_counts.json")
            with open(json_path, 'w') as json_file:
                json.dump(counts, json_file)
            save_dirs.append(image_dir)
        print(f"Failed for {num_fail} samples")
        return adv_images, save_dirs

# Updated main block to demonstrate both attackers
if __name__ == "__main__":
    # Get original embeddings
    paths = os.listdir('imagenet-mini')
    
    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    target_indices = torch.randint(0, 512 - 1, (25, )).to(device)

    # Save target points
    os.makedirs('keys', exist_ok=True)
    torch.save(target_indices, 'keys/target_points.pt')

    clip_attacker = CLIPAttacker(
        save_dir="adv_images_random_crop"
    )
    
    original_images = []

    paths = paths[:100]
    for path in paths:
        temp = Image.open(f"imagenet-mini/{path}")
        original_images.append(temp.copy())
        temp.close()

    # original_images = [Image.open(f"imagenet-mini/{path}") for path in paths]
    original_filenames = [os.path.basename(path) for path in paths]

    shuffle(original_images)
    # Attack and save with CLIP attacker
    adv_images_clip, saved_paths_clip = clip_attacker.attack_and_save(
        original_images,
        target_indices,
        original_filenames
    )
    
    print(f"Saved {len(saved_paths_clip)} CLIP adversarial images to {clip_attacker.save_dir}/")