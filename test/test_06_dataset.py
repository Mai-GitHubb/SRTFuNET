import sys
import cv2
import numpy as np
import torch
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import LateFusionDataset

def create_dummy_video(filename="dummy_test_video.mp4", width=224, height=224, frames=30, fps=30):
    """Creates a dummy video file with random noise."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for _ in range(frames):
        # Create a random noise frame
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print(f"Created dummy video: {filename}")

def test_late_fusion_dataset():
    """
    Tests the LateFusionDataset for correct output tensor shapes.
    """
    video_file = "dummy_test_video.mp4"
    create_dummy_video(filename=video_file)

    try:
        # Instantiate the dataset
        dataset = LateFusionDataset(video_paths=[video_file], labels=[1])

        # Fetch the first item
        data_item = dataset[0]

        # --- Assertions ---
        # 1. Spatial Tensor Shape
        expected_spatial_shape = (3, 299, 299)
        actual_spatial_shape = data_item['spatial'].shape
        assert actual_spatial_shape == expected_spatial_shape, f"Spatial tensor shape is {actual_spatial_shape}, but expected {expected_spatial_shape}"

        # 2. Temporal Tensor Shape
        expected_temporal_shape = (3, 16, 112, 112)
        actual_temporal_shape = data_item['temporal'].shape
        assert actual_temporal_shape == expected_temporal_shape, f"Temporal tensor shape is {actual_temporal_shape}, but expected {expected_temporal_shape}"

        # 3. Landmark Tensor Shape
        expected_landmark_shape = (1404,)
        actual_landmark_shape = data_item['landmark'].shape
        assert actual_landmark_shape == expected_landmark_shape, f"Landmark tensor shape is {actual_landmark_shape}, but expected {expected_landmark_shape}"
            
        # 4. Label value
        assert data_item['label'] == 1, f"Label is {data_item['label']}, but expected 1"

        print("\n--- Assertions ---")
        print(f"Spatial shape: {actual_spatial_shape} (Correct: {actual_spatial_shape == expected_spatial_shape})")
        print(f"Temporal shape: {actual_temporal_shape} (Correct: {actual_temporal_shape == expected_temporal_shape})")
        print(f"Landmark shape: {actual_landmark_shape} (Correct: {actual_landmark_shape == expected_landmark_shape})")
        print("Label value is correct.")
        
        print("\nPASS: LateFusionDataset outputs correct tensor shapes!")

    finally:
        # Clean up the dummy video file
        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"Cleaned up dummy video: {video_file}")

if __name__ == "__main__":
    test_late_fusion_dataset()
