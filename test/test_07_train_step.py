import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np

from dataset import LateFusionDataset
from late_fusion_model import LateFusionDeepfakeDetector

def create_dummy_video(filename, width=224, height=224, frames=30, fps=30):
    """Creates a dummy video file with random noise."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for _ in range(frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def test_training_step_and_gradient_flow():
    """
    Verifies the training loop, loss computation, and correct gradient flow.
    """
    # 1. Generate dummy data
    video_files = [f"dummy_{i}.mp4" for i in range(1, 5)]
    labels = [0, 1, 0, 1]
    for filename in video_files:
        create_dummy_video(filename)
    print(f"Created {len(video_files)} dummy videos.")

    try:
        # 2. Set up DataLoader
        dataset = LateFusionDataset(video_paths=video_files, labels=labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 3. Initialize Model, Optimizer, and Loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LateFusionDeepfakeDetector().to(device)
        model.train() # Set model to training mode

        # Filter for trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-4)
        
        criterion = nn.CrossEntropyLoss()
        print(f"Model and optimizer initialized on {device}.")

        # 4. Run ONE training step
        print("Running a single training step...")
        batch = next(iter(dataloader))

        # Move tensors to device
        spatial_img = batch['spatial'].to(device)
        temporal_seq = batch['temporal'].to(device)
        landmark_arr = batch['landmark'].to(device)
        label = batch['label'].to(device)

        # Zero gradients, forward pass, loss, backward, step
        optimizer.zero_grad()
        outputs = model(spatial_img, temporal_seq, landmark_arr)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        print(f"Loss computed: {loss.item()}")

        # 5. Gradient Assertions
        # Frozen part: SpatialExtractor's first conv layer
        frozen_grad = model.spatial_extractor.model.conv1.weight.grad
        assert frozen_grad is None, "Gradients found in frozen layer (SpatialExtractor)"
        print("Verified: No gradients in the frozen SpatialExtractor.")

        # Trainable part: Classifier's final layer
        trainable_grad = model.classifier[-1].weight.grad
        assert trainable_grad is not None, "No gradients found in trainable layer (Classifier)"
        print("Verified: Gradients are present in the trainable Classifier.")

        # Final Success Message
        print("\nPASS: Training loop executed, loss computed, and gradients flowed correctly to the trainable layers only!")

    finally:
        # 6. Cleanup
        for filename in video_files:
            if os.path.exists(filename):
                os.remove(filename)
        print(f"Cleaned up {len(video_files)} dummy videos.")

if __name__ == "__main__":
    test_training_step_and_gradient_flow()
