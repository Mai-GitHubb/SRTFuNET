import torch
import timm
import mediapipe

def test_pytorch_cuda():
    """
    Tests if PyTorch is installed and if CUDA is available.
    """
    print("--- Testing PyTorch and CUDA ---")
    try:
        if torch.cuda.is_available():
            print("PASS: PyTorch can access the GPU.")
            print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("FAIL: PyTorch cannot access the GPU.")
    except Exception as e:
        print(f"FAIL: An error occurred during PyTorch CUDA test: {e}")

def test_timm_models():
    """
    Tests if timm can list available Xception models.
    """
    print("\n--- Testing timm ---")
    try:
        xception_models = timm.list_models('xception*')
        if xception_models:
            print("PASS: timm can list Xception models.")
            # print("Available Xception models:")
            # for model in xception_models:
            #     print(f"- {model}")
        else:
            print("FAIL: timm could not list any Xception models.")
    except Exception as e:
        print(f"FAIL: An error occurred during timm test: {e}")

def test_mediapipe_face_mesh():
    """
    Tests if mediapipe's FaceMesh module can be initialized.
    """
    print("\n--- Testing mediapipe ---")
    try:
        with mediapipe.solutions.face_mesh.FaceMesh() as face_mesh:
            print("PASS: mediapipe.solutions.face_mesh.FaceMesh initialized successfully.")
    except Exception as e:
        print(f"FAIL: An error occurred during mediapipe test: {e}")

if __name__ == "__main__":
    test_pytorch_cuda()
    test_timm_models()
    test_mediapipe_face_mesh()
