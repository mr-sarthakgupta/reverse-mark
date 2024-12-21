import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # First Convolutional Layer
        # Input: 3 channels (RGB), Output: 16 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second Convolutional Layer
        # Input: 16 channels, Output: 32 feature maps, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First Linear Layer
        # Assuming input image size is 32x32, after two pooling layers it becomes 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        
        # Second Linear Layer
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv layer with batch norm, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv layer with batch norm, ReLU, and pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the feature maps
        x = x.view(-1, 32 * 8 * 8)
        
        # First linear layer with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Second linear layer (output layer)
        x = self.fc2(x)
        
        return x

# Example usage
def train_model():
    # Create model instance
    model = ConvNet(num_classes=10)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, criterion, optimizer

if __name__ == "__main__":
    model = ConvNet()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(model.device)