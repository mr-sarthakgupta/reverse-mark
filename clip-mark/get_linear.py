import torch
import torch.nn as nn
import os

def create_and_save_linear_layer(input_dim: int = 512, output_dim: int = 2):
    # Create the linear layer
    linear = nn.Linear(input_dim, output_dim)
    
    # Create directory if it doesn't exist
    os.makedirs("linear-layers", exist_ok=True)
    
    # Save the linear layer
    torch.save(linear.state_dict(), "linear-layers/layer_1.pth")
    print(f"Linear layer saved with input dim {input_dim} and output dim {output_dim}")

if __name__ == "__main__":
    create_and_save_linear_layer()