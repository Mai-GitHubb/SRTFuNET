import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from graph_stream import LandmarkExtractor

def test_landmark_extractor_output_shape():
    """
    Tests the output shape of the LandmarkExtractor model.
    """
    # Create an instance of the LandmarkExtractor
    extractor = LandmarkExtractor()
    extractor.eval()  # Set the model to evaluation mode

    # Create a dummy PyTorch tensor of shape (2, 1404)
    dummy_tensor = torch.randn(2, 1404)

    # Pass the dummy tensor through the extractor
    with torch.no_grad():
        output = extractor(dummy_tensor)

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output.shape}")

    # The expected shape is (batch_size, compressed_features)
    expected_shape = (2, 512)

    # Use an assert statement to enforce this shape
    assert output.shape == expected_shape, f"Output shape is {output.shape}, but expected {expected_shape}"

    # Print a success message if the assertion passes
    print("PASS: LandmarkExtractor output shape is correct.")

if __name__ == "__main__":
    test_landmark_extractor_output_shape()
