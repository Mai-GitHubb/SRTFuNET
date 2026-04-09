import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from spatial_stream import SpatialExtractor

def test_spatial_extractor_output_shape():
    """
    Tests the output shape of the SpatialExtractor model.
    """
    # Create an instance of the SpatialExtractor
    extractor = SpatialExtractor()
    extractor.eval()  # Set the model to evaluation mode

    # Create a dummy PyTorch tensor of shape (2, 3, 299, 299)
    dummy_tensor = torch.randn(2, 3, 299, 299)

    # Pass the dummy tensor through the extractor
    with torch.no_grad():
        output = extractor(dummy_tensor)

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output.shape}")

    # The expected shape is (batch_size, num_features) which is (2, 2048) for xception
    expected_shape = (2, 2048)

    # Use an assert statement to enforce this shape
    assert output.shape == expected_shape, f"Output shape is {output.shape}, but expected {expected_shape}"

    # Print a success message if the assertion passes
    print("PASS: SpatialExtractor output shape is correct.")

if __name__ == "__main__":
    test_spatial_extractor_output_shape()
