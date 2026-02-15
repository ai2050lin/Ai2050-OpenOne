
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionProjector(nn.Module):
    """
    Projects visual data (images) into the Logic Core's embedding manifold.
    Architecture: Simple CNN for MNIST (28x28 -> d_model)
    """
    def __init__(self, d_model=128):
        super().__init__()
        
        # 28x28x1 -> 14x14x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 14x14x32 -> 7x7x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Flatten: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [Batch, 1, 28, 28]
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 7 * 7 * 64)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x) # [Batch, d_model]
        
        # Normalize output to unit sphere? 
        # Logic embeddings might not be normalized, but keeping projection bounded is good.
        # Let's match the scale of logic embeddings (roughly sqrt(d_model) if normalized, but they are not strictly).
        # We will let the loss function handle the scaling alignment.
        
        return x

def create_vision_model(d_model=128):
    return VisionProjector(d_model=d_model)
