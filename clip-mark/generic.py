import os
import json
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from open_clip.src import open_clip as clip
from torchvision.transforms.functional import pil_to_tensor

def calculate_statistics(directory):
    total_difference = 0
    total_original_count = 0
    total_adv_count = 0
    file_count = 0
    original_counts = []
    adv_counts = []

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    original_count = data.get('original_count', 0)
                    adv_count = data.get('adversarial_count', 0)
                    total_difference += (original_count - adv_count)
                    total_original_count += original_count
                    total_adv_count += adv_count
                    original_counts.append(original_count)
                    adv_counts.append(adv_count)
                    file_count += 1

    if file_count == 0:
        return 0, 0, 0, 0, 0, 0, 0

    average_difference = total_difference / file_count
    average_original_count = total_original_count / file_count
    average_adv_count = total_adv_count / file_count

    std_dev_original = math.sqrt(sum((x - average_original_count) ** 2 for x in original_counts) / file_count)
    std_dev_adv = math.sqrt(sum((x - average_adv_count) ** 2 for x in adv_counts) / file_count)

    return average_difference, average_original_count, average_adv_count, std_dev_original, std_dev_adv, original_counts, adv_counts

directory = 'adv_images_l2'

average_difference, average_original_count, average_adv_count, std_dev_original, std_dev_adv, original_counts, adv_counts = calculate_statistics(directory)
print(f'Average difference: {average_difference}')
print(f'Average original count: {average_original_count}')
print(f'Average adversarial count: {average_adv_count}')
print(f'Standard deviation of original count: {std_dev_original}')
print(f'Standard deviation of adversarial count: {std_dev_adv}')

original_counts = np.array(original_counts)
adv_counts = np.array(adv_counts)

all_counts = np.concatenate((original_counts, adv_counts))

print(np.zeros(len(original_counts)).shape, np.ones(len(adv_counts)).shape)

print(np.concatenate((np.zeros(len(original_counts)), np.ones(len(adv_counts)))).shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_counts.reshape(-1, 1), np.concatenate((np.zeros(len(original_counts)), np.ones(len(adv_counts)))), test_size=0.2, random_state=42)


# Train the logistic regression model

model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print(f"Uisng thesholds only, we get:")

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


key = torch.load('keys_l2_400/target_points.pt')

print(key.shape)

def loss_fn(outputs, target_indices):
    softmax = nn.Softmax(dim=-1)
    softmax_outputs = softmax(outputs)
    dim1, dim2 = softmax_outputs.shape
    labels = torch.zeros(dim1, dim2)
    labels[:, target_indices] = 1
    if torch.isnan(softmax_outputs).any() or torch.isnan(labels).any():
        raise ValueError("NaN values found in the tensors")
    # return -1 * torch.norm(softmax_outputs - labels.cuda(), p=float('inf'), dim=-1)
    return -1 * torch.norm(softmax_outputs - labels.cuda(), p=2, dim=-1)

class CLIPFwd(nn.Module):
    def __init__(self):
        super(CLIPFwd, self).__init__()
        self.model, _, self.preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        
    def forward(self, x):
        return self.model.encode_image(self.preprocess(x))

def calculate_losses(directory, key):
    adv_losses = []
    orig_losses = []
    adv_file_count = 0
    orig_file_count = 0
    model = CLIPFwd()
    model.to("cuda:0")

    for subdir, _, files in os.walk(directory):
        for file in files:
            if 'adversarial' in file:
                file_path = os.path.join(subdir, file)
                inputs = Image.open(file_path)
                inputs = pil_to_tensor(inputs).unsqueeze(0).float().to("cuda:0") / 255
                outputs = model(inputs)
                loss = loss_fn(outputs, key)
                adv_losses.append(loss.item())
                adv_file_count += 1
            if 'original' in file:
                file_path = os.path.join(subdir, file)
                inputs = Image.open(file_path)
                inputs = pil_to_tensor(inputs).unsqueeze(0).float().to("cuda:0") / 255
                outputs = model(inputs)
                loss = loss_fn(outputs, key)
                orig_losses.append(loss.item())
                orig_file_count += 1

    return adv_losses, orig_losses, adv_file_count, orig_file_count


adv_losses, orig_losses, adv_file_count, orig_file_count = calculate_losses(directory, key)
print(f'adv_losses: {adv_losses} \n orig_losses: {orig_losses} \n adv_file_count: {adv_file_count} \n orig_file_count: {orig_file_count}')  

X_train, X_test, y_train, y_test = train_test_split(np.concatenate((adv_losses.numpy(), orig_losses.numpy())).reshape(-1, 1), np.concatenate((np.ones(len(adv_losses)), np.zeros(len(orig_losses))), test_size=0.2, random_state=42))

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Using losses only, we get:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)