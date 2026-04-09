import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from late_fusion_model import LateFusionDeepfakeDetector

def test_late_fusion_model_forward_pass():
    """
    Tests the end-to-end forward pass of the LateFusionDeepfakeDetector.
    """
    # Create an instance of the model
    model = LateFusionDeepfakeDetector()
    model.eval()  # Set the model to evaluation mode

    # Create dummy tensors for a batch size of 2
    spatial_input = torch.randn(2, 3, 299, 299)
    temporal_input = torch.randn(2, 3, 16, 112, 112)
    landmark_input = torch.randn(2, 1404)

    # Pass all three into the model
    with torch.no_grad():
        output = model(spatial_input, temporal_input, landmark_input)

    # Assert the output shape is exactly (2, 2)
    expected_shape = (2, 2)
    assert output.shape == expected_shape, f"Output shape is {output.shape}, but expected {expected_shape}"

    # Print a success message
    print("PASS: LateFusionDeepfakeDetector end-to-end forward pass is successful!")

if __name__ == "__main__":
    test_late_fusion_model_forward_pass()
