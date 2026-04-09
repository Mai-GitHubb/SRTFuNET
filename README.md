# SRTfuNET: Deepfake Detection Using Spatial-Temporal-Landmark Fusion

A deep learning framework for detecting deepfake videos using multi-stream neural networks that fuse spatial, temporal, and facial landmark information with late fusion and transformer-based cross-stream attention.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Performance Results](#performance-results)
3. [Folder Structure](#folder-structure)
4. [Project Architecture](#project-architecture)
5. [Environment Setup](#environment-setup)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

**SRTfuNET** (Spatial-Temporal-Landmark Fusion Network) is a comprehensive deepfake detection system that processes video data through three independent feature extraction streams:

- **Spatial Stream**: Extracts frame-level features using Xception CNN
- **Temporal Stream**: Captures motion dynamics using R(2+1)D video CNN
- **Landmark Stream**: Analyzes facial geometry using MediaPipe landmarks

These streams are fused using late fusion with transformer-based cross-stream attention, achieving high AUC scores on benchmark datasets.

**Key Features:**
- Multi-stream architecture for robust feature representation
- Transformer-based attention for cross-stream interaction
- Support for real-time inference via Gradio web interface
- Comprehensive evaluation metrics and visualization tools
- Advanced training pipeline with temperature scaling and threshold optimization
- Ensemble inference with test-time augmentation (TTA)

---

## Performance Results

### Test Set Evaluation (Ensemble of 3 Models)

The ensemble model achieves excellent performance on the test set with **518 videos** (Real: 178, Fake: 340):

| Metric | Score |
|--------|-------|
| **AUC** | **0.9705** ⭐ |
| **Accuracy** | **91.70%** |
| **Precision** | 0.9514 |
| **Recall** | 0.9206 |
| **F1-Score** | 0.9357 |
| **Real Accuracy** | 0.9101 |
| **Fake Accuracy** | 0.9206 |

### Confusion Matrix

```
                Predicted Real    Predicted Fake
  Actual Real |      162              16          (TN / FP)
  Actual Fake |      27               313         (FN / TP)
```

### Classification Report

```
              precision    recall  f1-score   support

        Real       0.86      0.91      0.88       178
        Fake       0.95      0.92      0.94       340

    accuracy                           0.92       518
   macro avg       0.90      0.92      0.91       518
```

### Ensemble Configuration

- **Models**: 3 checkpoint ensemble
  - `topk_e07_auc0.9736.pth` (Epoch 7 | Val AUC: 0.9736)
  - `topk_e10_auc0.9713.pth` (Epoch 10 | Val AUC: 0.9713)
  - `topk_e13_auc0.9680.pth` (Epoch 13 | Val AUC: 0.9680)
- **Temperature**: 1.3630 (calibrated)
- **Threshold**: 0.46 (optimized for balanced performance)
- **Test-Time Augmentation**: 1 pass per model

---

## Folder Structure

```
SRTfuNET/
├── app.py                          # Gradio web interface for inference
├── train.py                        # Main training script
├── inference.py                    # Batch inference pipeline
├── inf.py                          # Lightweight inference wrapper
├── evaluate_videos.py              # Video evaluation script
├── auto_eval.py                    # Automated evaluation with threshold sweep
├── threshold_sweep.py              # Threshold & temperature optimization
├── dataset.py                      # Data loading and augmentation
├── spatial_stream.py               # Xception-based spatial feature extractor
├── temporal_stream.py              # R(2+1)D-based temporal feature extractor
├── graph_stream.py                 # MediaPipe landmark feature extractor
├── late_fusion_model.py            # Main fusion model architecture
├── requirements.txt                # Python package dependencies
├── README.md                       # This file
├── README_CODE.md                  # Detailed code documentation
├── .gitignore                      # Git ignore patterns
├── checkpoints/                    # Pre-trained model weights
│   ├── topk_e07_auc0.9736.pth      # Ensemble model 1
│   ├── topk_e10_auc0.9713.pth      # Ensemble model 2
│   ├── topk_e13_auc0.9680.pth      # Ensemble model 3
│   ├── best_auc.pth
│   └── best_balanced.pth
├── data/                           # Dataset directory (not in repo)
│   ├── List_of_testing_videos.txt
│   ├── Celeb-real/                 # Real celebrity face videos
│   ├── Celeb-synthesis/            # AI-generated deepfake videos
│   └── YouTube-real/               # Real YouTube videos
├── fallback/                       # Fallback model weights (not in repo)
│   └── acc7198.pth
└── test/                           # Unit tests
    ├── test_01_environment.py
    ├── test_02_spatial.py
    ├── test_03_temporal.py
    ├── test_04_graph.py
    ├── test_05_fusion.py
    ├── test_06_dataset.py
    └── test_07_train_step.py
```

---

## Project Architecture

### Data Flow

```
Input Video
    ↓
[Frame Extraction (N=16 frames)]
    ↓
    ├─→ [Spatial Stream: Frame → Xception CNN] → (B, 2048)
    ├─→ [Temporal Stream: Video → R(2+1)D CNN] → (B, 512)
    └─→ [Landmark Stream: Landmarks → MLP] → (B, 512)
    ↓
[Project to Common Space: 512-d]
    ↓
[Cross-Stream Transformer Attention]
    ↓
[Classification Head + Temperature Scaling]
    ↓
Output: [Real/Fake Probability]
```

### Model Architecture

**Three Independent Streams:**

1. **Spatial Stream** (Xception)
   - Input: Single frame (3, 224, 224)
   - Output: 2048-d feature vector
   - Purpose: Extract appearance & compression artifacts

2. **Temporal Stream** (R(2+1)D-18)
   - Input: 16-frame video clip (3, 16, 224, 224)
   - Output: 512-d feature vector
   - Purpose: Capture motion dynamics & inconsistencies

3. **Landmark Stream** (MLP)
   - Input: 1405-d flattened landmarks (468 × 3 + 1 detection flag)
   - Output: 512-d feature vector
   - Purpose: Detect geometric impossibilities

**Fusion Strategy:**
- Late fusion: Features extracted independently
- Common space: Project all streams to 512-d
- Cross-stream attention: Transformer with 8 heads, 2 layers
- Classification: Binary head with configurable temperature scaling

---

## Environment Setup

### Prerequisites

- **Python**: 3.8 or higher (tested on 3.10)
- **CUDA**: 11.8+ (for GPU acceleration, highly recommended)
- **cuDNN**: 8.x (for CUDA support)
- **RAM**: 8 GB minimum (32 GB recommended)
- **GPU**: 4 GB minimum VRAM (24 GB recommended for training)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/Mai-GitHubb/SRTFuNET.git
cd SRTfuNET
```

#### 2. Create a Conda Environment

```bash
# Create environment with Python 3.10
conda create -n srtfunet python=3.10 -y

# Activate environment
conda activate srtfunet
```

#### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0` and `torchvision` - Deep learning framework
- `opencv-python` - Video processing
- `mediapipe` - Facial landmark detection (468 facial landmarks)
- `timm` - Pre-trained model zoo
- `gradio` - Web interface for easy testing
- `scikit-learn` - Evaluation metrics
- `pytorch-grad-cam` - Visualization (Grad-CAM)
- `numpy, pandas, matplotlib` - Data processing and visualization

#### 4. Verify Installation

Run the test suite to verify everything is working:

```bash
# Run all tests
python -m pytest test/ -v

# Or run individual tests
python test/test_01_environment.py
python test/test_02_spatial.py
python test/test_03_temporal.py
python test/test_04_graph.py
python test/test_05_fusion.py
python test/test_06_dataset.py
python test/test_07_train_step.py
```

#### 5. Prepare Pre-trained Checkpoints

Pre-trained checkpoints are included in the `checkpoints/` directory:
- `topk_e07_auc0.9736.pth` (Epoch 7 | Val AUC: 0.9736)
- `topk_e10_auc0.9713.pth` (Epoch 10 | Val AUC: 0.9713)
- `topk_e13_auc0.9680.pth` (Epoch 13 | Val AUC: 0.9680)

The ensemble of these three models achieves **0.9705 AUC** on the test set.

---

## Quick Start

### 1. Run the Web Interface (Inference)

```bash
conda activate srtfunet
python app.py
```

    Then open your browser to `http://localhost:7860` to upload and test videos.

    ### 2. Run Batch Inference on a Single Video

    ```bash
    python inf.py --video video.mp4 --checkpoint checkpoints/topk_e10_auc0.9713.pth
    ```

    ### 3. Evaluate on Dataset

    ```bash
    python evaluate_videos.py --data_dir data/ --checkpoint checkpoints/topk_e10_auc0.9713.pth
    ```

    ### 4. Run Automated Evaluation (Ensemble)

    ```bash
    python auto_eval.py
    ```

    This runs the complete pipeline:
    1. Threshold sweep & temperature calibration
    2. Inference on test set
    3. Generates comprehensive evaluation metrics

    ### 5. Optimize Threshold for New Model

    ```bash
    python threshold_sweep.py --checkpoint checkpoints/best_auc.pth
    ```

    ### 6. Train a New Model

    ```bash
    python train.py \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --data_dir ./data/ \
        --num_frames 16
    ```

    ---

    ## Configuration

    Key configuration parameters are typically found in the main scripts:

    | Parameter | Default | Description |
    |-----------|---------|-------------|
    | `num_frames` | `16` | Number of frames sampled per video |
    | `frame_size` | `224` | Spatial dimensions (224×224) |
    | `batch_size` | `32` | Training batch size |
    | `learning_rate` | `1e-4` | Adam optimizer learning rate |
    | `epochs` | `50` | Number of training epochs |
    | `TEMPERATURE` | `1.3630` | Temperature scaling (calibrated for ensemble) |
    | `THRESHOLD` | `0.46` | Classification threshold (optimized) |
    | `device` | `cuda` | Device (cuda or cpu) |

    ---

    ## System Requirements

    ### Minimum Specifications
    - **RAM**: 8 GB
    - **Storage**: 50 GB (for data and models)
    - **GPU**: 4 GB VRAM (CPU mode possible but slow)

    ### Recommended Specifications
    - **RAM**: 32 GB
    - **Storage**: 100 GB SSD
    - **GPU**: 24 GB VRAM (RTX 4090, A100, or equivalent)
    - **CPU**: Intel i7/i9 or AMD Ryzen 7/9
    - **OS**: Linux (Ubuntu 20.04+) or Windows 10+

    ---

    ## Troubleshooting

    ### GPU Not Detected

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

    If `False`, install CUDA-compatible PyTorch:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    ### MediaPipe Warnings

    MediaPipe GPU is disabled by default. To enable GPU acceleration:
    ```bash
    # On Linux/Mac
    export MEDIAPIPE_DISABLE_GPU=0

    # On Windows (PowerShell)
    $env:MEDIAPIPE_DISABLE_GPU=0
    ```

    ### Out of Memory (OOM) Errors

    - Reduce `batch_size` in training scripts (e.g., from 32 to 16)
    - Reduce `num_frames` from 16 to 8
    - Reduce `frame_size` from 224 to 192
    - Enable gradient checkpointing in model

    ### Video Format Issues

    Supported video formats: MP4, AVI, MOV, WebM
    ```bash
    # Convert video format with ffmpeg
    ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
    ```

    ### Import Errors

    Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt --upgrade
    ```

    ---

    ## Research Details

    ### Multi-Stream Architecture

    The three-stream design captures complementary visual and spatial information:

    1. **Spatial Stream (Xception)**
       - Detects compression artifacts from deepfake generation algorithms
       - Identifies unnatural skin texture and color bleeding
       - Learns CNN-based forensic features

    2. **Temporal Stream (R(2+1)D)**
       - Detects jittering and unnatural motion
       - Identifies lip-sync errors
       - Captures temporal inconsistencies in facial movements

    3. **Landmark Stream (MLP on MediaPipe)**
       - Detects geometric impossibilities (landmarks in impossible positions)
       - Identifies morphological deformations
       - Robust to video quality degradation

    ### Fusion Strategy

    - **Late Fusion**: Each stream is processed independently, reducing feature interdependency
    - **Transformer Attention**: Cross-stream interactions enable feature combination
    - **Temperature Scaling**: Calibration improves probability estimates

    ---

    ## Citation

    If you use this work in your research, please cite:

    ```bibtex
    @article{srtfunet2025,
      title={SRTfuNET: Deepfake Detection via Spatial-Temporal-Landmark Late Fusion},
      author={[Your Name]},
      year={2025}
    }
    ```

    ---

    ## License

    Please specify your license here (MIT, Apache 2.0, etc.)

    ---

    ## Support & Contributing

    For issues, questions, or contributions:
    - Open an issue on GitHub
    - Submit a pull request
    - Contact the maintainers

    ---

    ## Acknowledgments

    - **Xception Architecture**: Chollet, F. (2016). Xception: Deep Learning with Depthwise Separable Convolutions
    - **R(2+1)D**: Tran, D., et al. (2018). A Closer Look at Spatiotemporal Convolutions for Action Recognition
    - **MediaPipe**: Google MediaPipe Library
    - **timm**: PyTorch Image Models (Ross Wightman)

    **Last Updated**: March 2025
