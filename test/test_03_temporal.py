import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from temporal_stream import TemporalExtractor

def test_temporal_extractor_output_shape():
    """
    Tests the output shape of the TemporalExtractor model.
    """
    # Create an instance of the TemporalExtractor
    extractor = TemporalExtractor()
    extractor.eval()  # Set the model to evaluation mode

    # Create a dummy PyTorch tensor of shape (2, 3, 16, 112, 112)
    # Shape: (Batch, Channels, Frames, Height, Width)
    dummy_tensor = torch.randn(2, 3, 16, 112, 112)

    # Pass the dummy tensor through the extractor
    with torch.no_grad():
        output = extractor(dummy_tensor)

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output.shape}")

    # The expected shape is (batch_size, num_features) which is (2, 512) for r2plus1d_18
    expected_shape = (2, 512)

    # Use an assert statement to enforce this shape
    assert output.shape == expected_shape, f"Output shape is {output.shape}, but expected {expected_shape}"

    # Print a success message if the assertion passes
    print("PASS: TemporalExtractor output shape is correct.")

if __name__ == "__main__":
    test_temporal_extractor_output_shape()
