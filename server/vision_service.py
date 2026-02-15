
import os

import numpy as np
import torch
import torch.nn as nn

from models.vision_projector import VisionProjector


class VisionService:
    def __init__(self, d_model=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projector = VisionProjector(d_model=d_model).to(self.device)
        self.d_model = d_model
        
        # Load pre-trained weights if available (mocking if not for demo)
        self.weights_path = "models/vision_projector_weights.pt"
        if os.path.exists(self.weights_path):
            self.projector.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            print(f"VisionProjector weights loaded from {self.weights_path}")
        else:
            print("WARNING: VisionProjector weights not found. Using initialized weights for demo.")

    def project_image(self, image_data):
        """
        Projects a single 28x28 image into the manifold.
        image_data: np.array [1, 28, 28] or [28, 28]
        """
        self.projector.eval()
        with torch.no_grad():
            if len(image_data.shape) == 2:
                image_data = np.expand_dims(image_data, axis=0) # Add channel
            
            x = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 1, 28, 28]
            projection = self.projector(x) # [1, d_model]
            return projection.cpu().numpy().flatten()

    def get_mnist_anchors(self):
        """
        Generates 10 example projections for digits 0-9 for alignment visualization.
        """
        # In a real scenario, this would load representative samples from MNIST.
        # For this AGI demo, we'll generate structured 'prototype' digits if they don't exist.
        anchors = []
        for i in range(10):
            # Create a mock 28x28 image for the digit i (a simple bar or pattern)
            img = np.zeros((28, 28), dtype=np.float32)
            row = int((i / 10.0) * 28)
            img[max(0, row-2):min(28, row+2), :] = 1.0
            
            proj = self.project_image(img)
            anchors.append({
                "digit": i,
                "projection": proj.tolist(),
                "label": f"MNIST_{i}"
            })
        return anchors

vision_service = VisionService()
