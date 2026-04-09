# SRTfuNET: Comprehensive Code Documentation

Detailed explanation of each Python file and the dataset structure in the SRTfuNET deepfake detection framework.

---

## Table of Contents

1. [Core Model Files](#core-model-files)
2. [Training and Inference Files](#training-and-inference-files)
3. [Utility and Evaluation Scripts](#utility-and-evaluation-scripts)
4. [Data Handling](#data-handling)
5. [Testing Framework](#testing-framework)
6. [Quick Reference](#quick-reference)

---

## Core Model Files

### `spatial_stream.py`

**Purpose**: Extracts spatial (appearance) features from individual video frames using a pre-trained Xception CNN.

**Key Components**:
- **SpatialExtractor**: Xception-based feature extractor
  - Input: Single RGB frame (3, 224, 224)
  - Output: Feature vector (1, 2048)
  - Pre-trained on ImageNet, fine-tuned on deepfake dataset

**Architecture Details**:
- Uses `timm` library's Xception model (pre-trained on ImageNet)
- Freeze strategy:
  - Freeze: Layers 0-9 (early convolutional blocks)
  - Fine-tune: blocks.10, blocks.11, conv3, conv4, bn3, bn4 (late layers & head)
  - Rationale: Transfer learning from ImageNet while adapting to deepfakes

**Key Methods**:
- `__init__()`: Initializes Xception model, removes classification head
- `forward(x)`: Processes frame through CNN, returns 2048-d features

**Usage Example**:
```python
from spatial_stream import SpatialExtractor
extractor = SpatialExtractor()
frame = torch.randn(1, 3, 224, 224)  # Batch of 1 frame
features = extractor(frame)  # Output: (1, 2048)
```

**Why Spatial Stream Matters**:
- Captures compression artifacts from deepfake generation (JPEG artifacts, blockiness)
- Detects unnatural skin texture (color bleeding, texture discontinuities)
- Identifies facial asymmetries caused by AI generation
- Learns CNN-based forensic features that indicate synthetic content

---

### `temporal_stream.py`

**Purpose**: Extracts temporal (motion) features from video clips using R(2+1)D-18.

**Key Components**:
- **TemporalExtractor**: R(2+1)D-18 (separable 2D+3D convolutions)
  - Input: Video sequence (3, 16, 224, 224) where 16 = number of frames
  - Output: Feature vector (1, 512)
  - Pre-trained on Kinetics-400 action recognition dataset

**Architecture Details**:
- R(2+1)D separates spatial and temporal convolutions for computational efficiency
- Freeze strategy:
  - Freeze: layer1, early parts of layer2
  - Fine-tune: layer3, layer4 (late residual blocks for domain adaptation)
- FC classifier layer replaced with Identity to output raw features
- Output dimension reduced from 512 to match fusion space

**Key Methods**:
- `__init__()`: Initializes R(2+1)D model, removes classification head
- `forward(x)`: Processes video through 3D CNN, returns 512-d features

**Usage Example**:
```python
from temporal_stream import TemporalExtractor
extractor = TemporalExtractor()
video = torch.randn(1, 3, 16, 224, 224)  # 16-frame clip
features = extractor(video)  # Output: (1, 512)
```

**Why Temporal Stream Matters**:
- Detects unnatural motion patterns (jittering, flickering)
- Identifies lip-sync errors in deepfake speech videos
- Captures inconsistent eye movement patterns
- Learns temporal coherence violations that AI algorithms produce

---

### `graph_stream.py`

**Purpose**: Extracts facial landmark-based geometric features from MediaPipe landmarks.

**Key Components**:
- **LandmarkExtractor**: MLP that processes 468 MediaPipe facial landmarks
  - Input: Flattened landmark coordinates (1, 1405)
    - 468 MediaPipe landmarks × 3 (x, y, z coordinates) = 1404 values
    - +1 binary "face detected" flag = 1405 total
  - Output: Feature vector (1, 512)
  - Architecture: Two-layer MLP with skip connection

**Architecture**:
```
Input (1405-d)
    ↓
[Linear(1405 → 1024) + BatchNorm + ReLU + Dropout(0.2)]  [Main branch]
    ↓
[Linear(1024 → 512) + ReLU]
    ↓
[Skip Connection: Linear(1405 → 512)]
    ↓
[Features combined via: features + skip_features + LayerNorm]
    ↓
Output Features (512-d)
```

**MediaPipe Landmarks** (468 landmarks):
- Face contour: 10 landmarks (lower and upper jawline)
- Eyebrows: 10 landmarks (left and right)
- Eyes: 20 landmarks (left, right iris, left/right eye region)
- Nose: 10 landmarks (bridge, tip, nostrils)
- Mouth/Lips: 80 landmarks (outer lip contour, inner lips)
- Cheeks, forehead, jaw: Remaining landmarks
- Coordinates: (x, y, z) where z is depth confidence

**Face Detection Flag**:
- Value: 1.0 if face detected, 0.0 if landmark extraction failed
- Purpose: Handles cases where MediaPipe fails (occluded faces, profile views)
- When not detected: Landmark coordinates filled with 0.5 (neutral value)

**Key Methods**:
- `__init__(input_dim=1405)`: Initializes MLP with specified input dimension
- `forward(x)`: Processes landmarks through MLP with skip connection

**Usage Example**:
```python
from graph_stream import LandmarkExtractor
extractor = LandmarkExtractor(input_dim=1405)
landmarks = torch.randn(1, 1405)  # Flattened landmarks + detection flag
features = extractor(landmarks)  # Output: (1, 512)
```

**Why Landmark Stream Matters**:
- Detects morphological impossibilities (eyes too close, mouth too large)
- Identifies incorrect landmark positions (distorted facial geometry)
- Captures non-rigid deformation patterns produced by GANs
- Robust to video compression and quality variations
- Complements spatial/temporal streams by focusing on geometric constraints

---

### `late_fusion_model.py`

**Purpose**: Combines all three feature streams using late fusion with transformer-based cross-stream attention.

**Key Components**:
- **LateFusionDeepfakeDetector**: Main model combining spatial, temporal, and landmark streams

**Architecture Overview**:
```
Spatial Stream (2048-d)    Temporal Stream (512-d)    Landmark Stream (1405-d)
        ↓                                ↓                         ↓
[Linear: 2048→512]      [Skip: 512→512]          [LandmarkExtractor]
        ↓                                ↓                         ↓
[BatchNorm + ReLU]      [BatchNorm + ReLU]       [Output: 512-d]
        ↓                                ↓                         ↓
      Spatial Features       Temporal Features      Landmark Features
            (B, 512)              (B, 512)                (B, 512)
                ↓                         ↓                         ↓
                └─────→ [Stack] ─────────┘ (B, 3, 512)
                             ↓
                 [Transformer Encoder]
                 (8 heads, 2 layers, Pre-LN)
                             ↓
                 [Output: (B, 3, 512)]
                             ↓
                 [Mean Pooling: (B, 512)]
                             ↓
                 [Classification Head]
                 [Linear(512 → 2)]
                             ↓
                  [Logits: (B, 2)]
                  (Real/Fake scores)
```

**Key Features**:

1. **Late Fusion Design**:
   - Each stream processes features independently
   - Features combined at decision level (not at early stages)
   - Reduces feature interdependency, allows specialized learning

2. **Cross-Stream Transformer Attention**:
   - Bidirectional attention across all three streams
   - 8 attention heads for multi-headed information fusion
   - 2 encoder layers for hierarchical feature interaction
   - Pre-layer normalization for training stability
   - Allows model to learn which stream features are most informative for each sample

3. **Classification Head**:
   - Binary classifier (Real vs Fake)
   - Logit-based output (before softmax)
   - Temperature scaling support for probability calibration

**Key Methods**:
- `__init__()`: Initializes all three streams, transformer, and classification head
- `forward(spatial, temporal, landmarks)`: Main forward pass
  - Projects all streams to 512-d
  - Stacks features into (B, 3, 512) tensor
  - Processes through transformer
  - Pools and classifies output
- `_extract_features()`: Internal method for feature extraction

**Temperature Scaling**:
- Applied during inference: `logits / temperature`
- Purpose: Calibrate confidence estimates
- Calibrated value for ensemble: `T = 1.3630`
- Lower temperature → sharper predictions, higher confidence
- Higher temperature → softer predictions, lower confidence

**Usage Example**:
```python
from late_fusion_model import LateFusionDeepfakeDetector
model = LateFusionDeepfakeDetector()

spatial = torch.randn(1, 3, 224, 224)        # Single frame
temporal = torch.randn(1, 3, 16, 224, 224)  # 16-frame clip
landmarks = torch.randn(1, 1405)             # Flattened landmarks

logits = model(spatial, temporal, landmarks)  # Output: (1, 2) logits
probabilities = torch.softmax(logits, dim=1) # Convert to probabilities
```

---

## Training and Inference Files

### `train.py`

**Purpose**: Main training loop for the SRTfuNET deepfake detector.

**Key Functionality**:
1. **Data Loading**:
   - Loads training and validation datasets from directories
   - Uses LateFusionDataset for video loading and preprocessing
   - Multi-worker DataLoader for parallel processing

2. **Training Loop**:
   - Forward pass through all three streams
   - Computes binary cross-entropy loss (or focal loss)
   - Backward pass with gradient accumulation
   - Learning rate warmup and decay
   - Validation after each epoch

3. **Optimization**:
   - Optimizer: Adam with configurable learning rate (default: 1e-4)
   - Selective unfreezing: Only fine-tune late layers
   - Learning rate scheduling: Warmup + exponential decay
   - Early stopping based on validation AUC

4. **Checkpointing**:
   - Saves best model by AUC score
   - Saves best model by balanced F1-score
   - Saves periodic checkpoints
   - Enables resuming training from checkpoint

5. **Evaluation**:
   - Computes AUC-ROC, Accuracy, Precision, Recall, F1-Score
   - Logs metrics to console and TensorBoard
   - Per-epoch validation on entire validation set

**Command-line Arguments**:
```bash
--epochs=50                    # Number of training epochs
--batch_size=32               # Batch size for training
--learning_rate=1e-4          # Adam optimizer learning rate
--num_workers=4               # Number of data loader workers
--data_dir='./data/'          # Dataset directory
--checkpoint=None             # Path to pre-trained model (optional)
--num_frames=16               # Frames per video
--seed=42                     # Random seed for reproducibility
```

**Usage**:
```bash
python train.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_dir ./data/ \
    --num_frames 16
```

**Output**:
- Best model checkpoint saved to `checkpoints/best_auc.pth`
- Training logs with per-epoch metrics
- TensorBoard logs for visualization

---

### `inference.py`

**Purpose**: Batch inference pipeline for evaluating multiple videos.

**Key Functionality**:
1. **Model Loading**:
   - Loads pre-trained model from checkpoint
   - Supports single or ensemble models
   - Loads to GPU if available

2. **Video Processing**:
   - Reads video file with OpenCV
   - Extracts frames and landmarks
   - Handles variable-length videos

3. **Inference**:
   - Extracts features from all three streams
   - Passes through model
   - Applies temperature scaling if specified
   - Applies threshold for binary classification

4. **Output**:
   - Probability score for each video
   - Classification (Real/Fake)
   - Confidence metrics
   - Inference time statistics

**Command-line Arguments**:
```bash
--checkpoint='checkpoints/topk_e10_auc0.9713.pth'
--video_path='path/to/video.mp4'
--temperature=1.3630          # Temperature scaling
--threshold=0.46              # Classification threshold
--tta=1                       # Test-time augmentation passes
--save_results=True           # Save results to JSON
```

**Usage**:
```bash
python inference.py \
    --checkpoint checkpoints/topk_e10_auc0.9713.pth \
    --video_path test_video.mp4 \
    --temperature 1.3630 \
    --threshold 0.46
```

---

### `inf.py`

**Purpose**: Lightweight inference wrapper for quick testing and deployment.

**Key Differences from inference.py**:
- Simpler, more optimized for production use
- Minimal dependencies
- Single-file inference
- Supports batch processing

**Main Functions**:
- `load_model()`: Loads model and weights
- `infer_video()`: Run inference on single video
- `batch_infer()`: Run inference on multiple videos

---

### `app.py`

**Purpose**: Gradio web interface for user-friendly video testing and inference.

**Features**:
1. **Web Interface**:
   - Upload video files (MP4, AVI, MOV, WebM)
   - Display results with visualization
   - Real-time processing feedback
   - Download results

2. **Visualization**:
   - Output probability (Real vs Fake)
   - Confidence score
   - Classification prediction
   - Grad-CAM attention maps (optional)

3. **Model Ensemble**:
   - Loads all 3 pre-trained checkpoints
   - Performs voting ensemble inference
   - Outputs ensemble prediction confidence

4. **Configuration**:
   - Calibrated temperature: 1.3630
   - Optimized threshold: 0.46
   - Test-time augmentation support

**Starting the Web Interface**:
```bash
python app.py
```

Then open browser to: `http://localhost:7860`

---

## Utility and Evaluation Scripts

### `evaluate_videos.py`

**Purpose**: Comprehensive evaluation on entire dataset.

**Key Functionality**:
- Evaluates all videos in specified directory
- Generates confusion matrix
- Computes classification metrics (precision, recall, F1)
- Generates per-class accuracy statistics
- Exports results to CSV/JSON

**Usage**:
```bash
python evaluate_videos.py \
    --data_dir ./data/ \
    --checkpoint checkpoints/topk_e10_auc0.9713.pth \
    --temperature 1.3630 \
    --threshold 0.46
```

---

### `threshold_sweep.py`

**Purpose**: Optimize classification threshold and temperature calibration.

**Key Functionality**:
1. **Temperature Calibration**:
   - Searches for optimal temperature value
   - Optimizes calibration error (ECE)
   - Enables accurate confidence estimates

2. **Threshold Optimization**:
   - Sweeps classification threshold from 0.0 to 1.0
   - Tracks accuracy, precision, recall, F1
   - Identifies balanced operating point
   - Outputs ROC curve

3. **Output**:
   - Calibrated temperature value
   - Optimal threshold for balanced performance
   - Visualization plots

**Usage**:
```bash
python threshold_sweep.py \
    --checkpoint checkpoints/best_auc.pth \
    --val_dir ./data/val/ \
    --num_thresholds 100
```

---

### `auto_eval.py`

**Purpose**: Automated evaluation pipeline combining all evaluation steps.

**Workflow**:
1. Finds all checkpoints in `checkpoints/` directory
2. For each checkpoint:
   - Runs threshold sweep (calibration)
   - Runs inference on test set
   - Collects evaluation metrics
3. Generates comparison report
4. Identifies best performing checkpoint/configuration

**Usage**:
```bash
python auto_eval.py
```

**Output**:
- Summary table comparing all checkpoints
- Best configuration identification
- Detailed metrics for each model

---

## Data Handling

### `dataset.py`

**Purpose**: PyTorch Dataset class for loading, preprocessing, and augmenting video data.

**Key Classes**:
- **LateFusionDataset**: Inherits from torch.utils.data.Dataset

**Dataset Operations**:

1. **Video Reading**:
   - OpenCV reads video file
   - Extracts all frames from video

2. **Frame Sampling**:
   - Uniformly samples N frames (default: 16) from entire video
   - Ensures consistent temporal sampling across variable-length videos
   - Formula: `frame_indices = linspace(0, total_frames, N)`

3. **Face Detection & Landmark Extraction**:
   - MediaPipe Holistic extracts 468 landmarks per frame
   - Coordinates: (x, y, z) where z is depth confidence
   - Detection flag: 1.0 if successful, 0.0 if failed

4. **Preprocessing**:
   - Normalizes coordinates to [0, 1] range
   - Handles cases where landmarks not detected
   - Stacks frames into video tensor (3, 16, 224, 224)

5. **Data Augmentation** (Training Only):

   **For Real Videos** (light augmentation):
   - Horizontal flip: p=0.5
   - Color jitter: p=0.6, brightness/contrast/saturation
   - Random rotation: ±8°, p=0.4
   - Gaussian noise: p=0.3, σ=0.02
   - Random grayscale: p=0.1
   - Rationale: Minimal distortion preserves genuine authenticity signals

   **For Fake Videos** (standard augmentation):
   - Horizontal flip: p=0.5
   - Color jitter: p=0.4
   - Gaussian noise: p=0.2
   - Rationale: Helps model generalize to various deepfake generation methods

**Key Methods**:
- `__init__(video_paths, labels, is_training=False)`: Initialize dataset
- `__len__()`: Returns total number of videos
- `__getitem__(idx)`: Returns tuple of (spatial, temporal, landmarks, label)
  - spatial: Single frame (3, 224, 224)
  - temporal: 16-frame video (3, 16, 224, 224)
  - landmarks: Flattened landmarks (1405,)
  - label: 0 for real, 1 for fake
- `_load_video(path)`: Read video and sample frames
- `_extract_landmarks(frames)`: Extract MediaPipe landmarks for each frame
- `_apply_augmentation(frames, is_real)`: Apply transformations based on class

**Usage Example**:
```python
from dataset import LateFusionDataset
video_paths = ['video1.mp4', 'video2.mp4']
labels = [0, 1]  # 0=real, 1=fake
dataset = LateFusionDataset(video_paths, labels, is_training=True)

# Load a sample
spatial, temporal, landmarks, label = dataset[0]
print(f"Spatial shape: {spatial.shape}")      # (3, 224, 224)
print(f"Temporal shape: {temporal.shape}")    # (3, 16, 224, 224)
print(f"Landmarks shape: {landmarks.shape}")  # (1405,)
print(f"Label: {label}")                      # 0 or 1
```

---

## Testing Framework

### `test/test_01_environment.py`

**Purpose**: Verify installation and environment setup.

**Tests**:
- Python version check (3.8+)
- PyTorch installation and CUDA availability
- Required library imports
- Device accessibility

---

### `test/test_02_spatial.py`

**Purpose**: Unit tests for spatial stream (Xception).

**Tests**:
- Model initialization
- Forward pass with dummy input
- Output shape verification
- Feature extraction correctness

---

### `test/test_03_temporal.py`

**Purpose**: Unit tests for temporal stream (R(2+1)D).

**Tests**:
- Model initialization
- 3D convolution forward pass
- Video processing pipeline
- Output feature dimensions

---

### `test/test_04_graph.py`

**Purpose**: Unit tests for landmark stream (MLP).

**Tests**:
- Landmark dimension handling (1405-d)
- MLP forward pass
- Skip connection functionality
- Feature extraction

---

### `test/test_05_fusion.py`

**Purpose**: Unit tests for full fusion model.

**Tests**:
- All three streams combined
- Transformer attention mechanism
- Classification head
- End-to-end forward pass

---

### `test/test_06_dataset.py`

**Purpose**: Unit tests for data loading and augmentation.

**Tests**:
- Video loading
- Frame sampling
- Landmark extraction
- Augmentation pipeline
- Data shape consistency

---

### `test/test_07_train_step.py`

**Purpose**: Unit tests for training loop.

**Tests**:
- Forward-backward pass
- Loss computation
- Gradient flow
- Optimization step
- Checkpoint saving/loading

---

## Quick Reference

### Model File Dependencies

```
late_fusion_model.py
├── spatial_stream.py
├── temporal_stream.py
└── graph_stream.py

train.py
├── late_fusion_model.py
└── dataset.py

inference.py
├── late_fusion_model.py
└── dataset.py

app.py
└── late_fusion_model.py
```

### Key Input/Output Shapes

| Module | Input Shape | Output Shape |
|--------|-----------|--------------|
| SpatialExtractor | (B, 3, 224, 224) | (B, 2048) |
| TemporalExtractor | (B, 3, 16, 224, 224) | (B, 512) |
| LandmarkExtractor | (B, 1405) | (B, 512) |
| LateFusionDeepfakeDetector | Three streams above | (B, 2) logits |

### Feature Extraction Pipeline

```
Raw Video (.mp4) 
    ↓
[Frame Extraction (16 frames)]
    ├→ [Spatial]: Frame → Xception → 2048-d
    ├→ [Temporal]: Video → R(2+1)D → 512-d  
    └→ [Landmark]: Frames → MediaPipe → 1405-d → MLP → 512-d
    ↓
[Fusion Transformer]
    ↓
[Classification Head]
    ↓
Output: [P(Real), P(Fake)]
```
[Linear(1024 → 512) + ReLU]
    ↓
Output Features (512-d)
    
Note: Skip connection from input to output via Linear(1405 → 512)
      Features combined via residual add + LayerNorm
```

**MediaPipe Landmarks**:
- 468 3D facial landmarks covering:
  - Face contour (lower and upper)
  - Eyebrows, eyes, nose
  - Mouth, lips
  - Cheeks, jaw
- Coordinates: (x, y, z) where z is depth confidence

**Face Detection Flag**:
- Added to handle cases where MediaPipe fails to detect landmarks
- When face not detected: coordinates filled with 0.5 (neutral value)
- Flag value: 1.0 (detected) or 0.0 (not detected)

**Usage Example**:
```python
from graph_stream import LandmarkExtractor
extractor = LandmarkExtractor(input_dim=1405)
landmarks = torch.randn(1, 1405)  # Flattened landmarks + detection flag
features = extractor(landmarks)  # Output: (1, 512)
```

**Why Landmark Stream Matters**:
- Detects morphological impossibilities (distorted facial geometry)
- Identifies incorrect landmark positions (eyes too close, mouth too large)
- Captures non-rigid deformation patterns
- Robust to video compression and quality variations

---

### `late_fusion_model.py`

**Purpose**: Combines all three feature streams using late fusion with transformer attention.

**Key Components**:
- **LateFusionDeepfakeDetector**: Main model combining spatial, temporal, and landmark streams

**Architecture**:
```
Spatial Stream (2048-d)     Temporal Stream (512-d)    Landmark Stream (1405-d)
        ↓                                ↓                         ↓
[Linear: 2048→512]      [Skip: 512→512]              [LandmarkExtractor]
        ↓                                ↓                         ↓
[Project to 512-d]      [Project to 512-d]           [Project to 512-d]
        ↓                                ↓                         ↓
      [Spatial]              [Temporal]                   [Landmark]
        ↓                                ↓                         ↓
                    [Cross-Stream Transformer]
                  (8-head attention, 2 layers)
                         ↓
                  [3 Output Tokens]
                         ↓
                    [Aggregation]
                         ↓
                    [Class Head]
                         ↓
                  [Real/Fake Logit]
```

**Key Features**:
- **Late Fusion**: Features extracted independently, combined at decision level
- **Cross-Stream Attention**: Transformer learns feature interactions
  - 8 attention heads
  - 2 encoder layers
  - Bidirectional attention across streams
  - Pre-layer normalization for training stability
- **Classification Head**: Binary classifier with configurable temperature

**Forward Pass**:
1. Extract features from all three streams in parallel
2. Project to common dimension (512-d)
3. Pass through 2-layer transformer with cross-attention
4. Aggregate outputs from transformer
5. Pass through classifier head

**Usage Example**:
```python
from late_fusion_model import LateFusionDeepfakeDetector
model = LateFusionDeepfakeDetector()
spatial = torch.randn(1, 3, 224, 224)        # Single frame
temporal = torch.randn(1, 3, 16, 224, 224)  # 16-frame clip
landmarks = torch.randn(1, 1405)             # Flattened landmarks

output = model(spatial, temporal, landmarks)  # Output: (1, 2) logits
```

---

## Training and Inference Files

### `dataset.py`

**Purpose**: Data loading, preprocessing, and augmentation for training.

**Key Classes**:
- **LateFusionDataset**: PyTorch Dataset for video data
  - Inherits from `torch.utils.data.Dataset`
  - Handles video reading, frame sampling, landmark extraction
  - Applies augmentation (training only)

**Dataset Operations**:
1. **Video Reading**: OpenCV reads video file and extracts frames
2. **Frame Sampling**: Uniformly samples N=16 frames from video
3. **Face Detection**: MediaPipe Holistic extracts 468 landmarks
4. **Frame Preprocessing**:
   - Resize to 224×224 (Xception/R(2+1)D standard)
   - Normalize: (x - mean) / std

**Data Augmentation** (Training Only):

For **Real** Videos (toned down to preserve genuine signal):
- Horizontal flip: p=0.5
- Color jitter: p=0.6, brightness/contrast/saturation
- Random rotation: ±8°, p=0.4
- Gaussian noise: p=0.3, σ=0.02
- Random grayscale: p=0.1

For **Fake** Videos (standard augmentation):
- Horizontal flip: p=0.5
- Color jitter: p=0.4 (milder)
- Gaussian noise: p=0.2 (milder)

**Rationale**: Real faces need minimal augmentation to preserve authenticity signals. Too much distorts genuine face features.

**Key Methods**:
- `__len__()`: Returns dataset size
- `__getitem__(idx)`: Returns (spatial, temporal, landmarks, label) tuple
- `_load_video()`: Read video and sample frames
- `_extract_landmarks()`: Extract MediaPipe landmarks for each frame
- `_apply_augmentation()`: Apply transformations

**Usage Example**:
```python
from dataset import LateFusionDataset
video_paths = ['video1.mp4', 'video2.mp4']
labels = [0, 1]  # 0=real, 1=fake
dataset = LateFusionDataset(video_paths, labels, is_training=True)
spatial, temporal, landmarks, label = dataset[0]
```

---

### `train.py`

**Purpose**: Main training loop for the deepfake detector.

**Key Functions**:

1. **Training Pipeline**:
   - Data loading with multi-worker DataLoader
   - Loss computation (binary cross-entropy or focal loss)
   - Backward pass with gradient accumulation
   - Evaluation on validation set
   - Model checkpointing

2. **Configuration Parameters**:
   - `--epochs`: Number of training epochs
   - `--batch_size`: Batch size for training
   - `--learning_rate`: Adam optimizer learning rate
   - `--checkpoint`: Path to pre-trained model
   - `--data_dir`: Dataset directory

3. **Training Strategy**:
   - Selective unfreezing of pre-trained networks
   - Warmup phase for learning rate
   - Exponential learning rate decay
   - Early stopping based on validation AUC
   - Model checkpointing for best validation metrics

4. **Evaluation Metrics**:
   - AUC-ROC (Area Under ROC Curve)
   - Accuracy, Precision, Recall, F1-Score
   - Logs to console and TensorBoard

**Output**:
- Best model saved to `checkpoints/best_auc.pth`
- Training logs with epoch statistics
- Validation metrics per epoch

**Example Usage**:
```bash
python train.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_dir ./data/ \
    --checkpoint checkpoints/topk_epoch10_auc0.9672.pth
```

---

### `inference.py`

**Purpose**: Batch inference on video files.

**Key Functions**:

1. **Inference Pipeline**:
   - Load pre-trained model
   - Read video files
   - Extract spatial, temporal, landmark features
   - Forward pass through model
   - Apply temperature scaling and threshold
   - Output real/fake predictions with confidence

2. **Temperature Calibration**:
   - Temperature T controls output confidence
   - T < 1.0: More confident predictions
   - T > 1.0: More uncertain predictions
   - Default: T=1.4498 (empirically tuned)

3. **Threshold Configuration**:
   - Default threshold: 0.5 (no bias)
   - Can be adjusted based on operating point
   - Default in `app.py`: 0.64

**Example Usage**:
```bash
python inference.py \
    --video_path video.mp4 \
    --checkpoint checkpoints/topk_epoch10_auc0.9672.pth
```

---

### `inf.py`

**Purpose**: Lightweight inference wrapper (simplified inference interface).

**Key Functions**:
- Minimal setup for quick inference
- Single video processing
- Returns binary prediction (real/fake) and confidence
- Used by web interface and batch processing

---

### `app.py`

**Purpose**: Web interface for real-time inference using Gradio.

**Key Features**:
- **Gradio Interface**: User-friendly web UI
- **Video Upload**: Users upload MP4 or WebM files
- **Real-Time Processing**: Process video and display results
- **Grad-CAM Visualization**: Visualize which image regions influenced prediction
- **Output Display**:
  - Classification result (Real/Fake)
  - Confidence score
  - Attention heatmap overlay on frames

**Model Configuration in app.py**:
```python
CHECKPOINT_PATH = "checkpoints/topk_epoch10_auc0.9672.pth"
TEMPERATURE = 1.4498
THRESHOLD = 0.6400
```

**Grad-CAM Components**:
- **SpatialWrapper**: Focuses gradient computation on spatial stream input
- **TemporalWrapper**: Focuses gradient computation on temporal stream input
- **LandmarkWrapper**: Focuses gradient computation on landmark stream input
- Overlaid on frames to show decision influences

**Running the Web App**:
```bash
python app.py
# Open browser to http://localhost:7860
```

---

## Data Handling

### `dataset.py` (Detailed)

**Data Flow**:
```
Video File (MP4)
    ↓
[OpenCV Video Read]
    ↓
[Frame Extraction: Sample 16 frames uniformly]
    ↓
├─→ [224×224 resize + norm] → Spatial input
├─→ [Stack 16 frames + temporal augmentation] → Temporal input
└─→ [MediaPipe landmark detection + augmentation] → Landmark input
    ↓
[Return: (spatial, temporal, landmarks, label)]
```

**Special Handling**:
- **Failed Landmark Detection**: Fill with 0.5 (neutral) + set flag to 0.0
- **Short Videos**: Repeat frames if video has <16 frames
- **Corrupted Videos**: Skip and log error
- **Augmentation Seed**: Set per sample for reproducibility

---

## Evaluation and Utilities

### `evaluate_videos.py`

**Purpose**: Comprehensive evaluation on test dataset.

**Functionality**:
- Loads all videos from dataset
- Runs inference on each video
- Computes evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC, AP (Average Precision)
  - Confusion matrix
  - Per-category metrics (real vs fake separately)
- Generates visualizations (ROC curve, confusion matrix)
- Outputs results to CSV and JSON

**Usage**:
```bash
python evaluate_videos.py \
    --data_dir ./data/ \
    --checkpoint checkpoints/topk_epoch10_auc0.9672.pth \
    --output results.json
```

---

### `auto_eval.py`

**Purpose**: Automated evaluation pipeline.

**Features**:
- Periodic evaluation during training
- Automatic metric logging
- Comparison across checkpoints
- Early stopping trigger

---

### `threshold_sweep.py`

**Purpose**: Find optimal classification threshold.

**Functionality**:
- Sweeps threshold from 0.0 to 1.0
- Computes metrics at each threshold:
  - TNR (True Negative Rate / Specificity)
  - TPR (True Positive Rate / Sensitivity)
  - F1-Score, Balanced Accuracy
- Finds threshold maximizing chosen metric
- Plots threshold vs metric curve

**Why Threshold Optimization Matters**:
- Default threshold=0.5 assumes balanced cost of FP/FN
- For fraud detection: minimize FN (catch all fakes)
- For authentication: minimize FP (minimal false alarms)
- Optimal threshold depends on application

**Usage**:
```bash
python threshold_sweep.py \
    --checkpoint checkpoints/topk_epoch10_auc0.9672.pth \
    --val_dir data/ \
    --metric f1
```

---

### `graph_stream.py` (Explained above)

**Additional utilities in graph_stream.py**:
- Landmark filtering: Remove invalid landmarks
- Landmark normalization: Normalize to face bounding box
- Distance metrics: Compute inter-landmark distances for shape analysis

---

## Dataset Structure

### Directory Layout

```
data/
├── List_of_testing_videos.txt      # Names of test videos
├── Celeb-real/                     # High-quality real celebrity videos
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── Celeb-synthesis/                # AI-generated deepfake videos
│   ├── fake_001.mp4
│   ├── fake_002.mp4
│   └── ...
└── YouTube-real/                   # YouTube real videos (broader diversity)
    ├── youtube_001.mp4
    ├── youtube_002.mp4
    └── ...
```

### Dataset Split Convention

**Typical splits** (adjust based on your needs):
- **Training**: 80% of Celeb-real + 80% of Celeb-synthesis
- **Validation**: 10% each
- **Test**: 10% each (or separate YouTube-real set)

### Data Characteristics

| Category | Count | Duration | Source | Characteristics |
|----------|-------|----------|--------|-----------------|
| Celeb-real | ~32 | 2-10 min | High-quality recordings | Clear faces, controlled lighting |
| Celeb-synthesis | ~32 | 2-10 min | FaceSwap, Wav2Lip, etc. | Various deepfake artifacts |
| YouTube-real | ~300+ | 5-30 min | YouTube videos | Diverse backgrounds, lighting, angles |

### Video Format Requirements

- **Codec**: H.264 (MP4) preferred, H.265/WebM supported
- **Frame Rate**: 24-30 FPS
- **Resolution**: 480p-1080p (auto-resized to 224×224)
- **Duration**: Any (frames dynamically sampled)
- **Bitrate**: 1-10 Mbps

### Label Convention

```
1 = Fake (deepfake)
0 = Real (authentic)
```

---

## Testing Framework

### Test Files Location

All tests are in the `test/` directory:
```
test/
├── test_01_environment.py          # Environment setup verification
├── test_02_spatial.py              # Spatial stream tests
├── test_03_temporal.py             # Temporal stream tests
├── test_04_graph.py                # Landmark stream tests
├── test_05_fusion.py               # Fusion model tests
├── test_06_dataset.py              # Dataset loader tests
└── test_07_train_step.py           # Training step tests
```

### Test Coverage

1. **test_01_environment.py**
   - GPU detection
   - Package imports
   - CUDA/cuDNN versions
   - Device availability

2. **test_02_spatial.py**
   - SpatialExtractor initialization
   - Forward pass with correct input shape
   - Output dimensions validation
   - Gradient flow verification

3. **test_03_temporal.py**
   - TemporalExtractor initialization
   - Forward pass with video tensor
   - Output feature shape
   - Temporal consistency

4. **test_04_graph.py**
   - LandmarkExtractor initialization
   - Forward pass with landmark tensor
   - Landmark processing
   - Missing landmark handling

5. **test_05_fusion.py**
   - LateFusionDeepfakeDetector initialization
   - Multi-stream forward pass
   - Output validation
   - Transformer attention verification

6. **test_06_dataset.py**
   - Dataset initialization
   - Sample loading and preprocessing
   - Augmentation application
   - Label handling

7. **test_07_train_step.py**
   - Loss computation
   - Backward pass verification
   - Gradient updates
   - Optimizer states

### Running Tests

```bash
# Run all tests
pytest test/ -v

# Run specific test
pytest test/test_02_spatial.py -v

# Run with coverage report
pytest test/ --cov=. --cov-report=html

# Run tests matching pattern
pytest test/ -k spatial -v
```

---

## File Dependencies

```
late_fusion_model.py (main model)
    ├── depends on: spatial_stream.py
    ├── depends on: temporal_stream.py
    └── depends on: graph_stream.py

train.py (training)
    ├── depends on: late_fusion_model.py
    └── depends on: dataset.py

inference.py (inference)
    ├── depends on: late_fusion_model.py
    └── depends on: dataset.py (for preprocessing)

app.py (web interface)
    ├── depends on: late_fusion_model.py
    ├── depends on: inference.py
    └── depends on: pytorch_grad_cam (visualization)

evaluate_videos.py (evaluation)
    ├── depends on: inference.py
    └── depends on: scikit-learn (metrics)
```

---

## Version Information

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Language runtime |
| PyTorch | 2.0+ | Deep learning |
| TorchVision | 0.15+ | Computer vision utils |
| TIMM | 0.9+ | Pre-trained models |
| MediaPipe | 0.10+ | Landmark detection |
| OpenCV | 4.8+ | Video processing |
| Gradio | 3.50+ | Web UI |

---

## Performance Notes

- **Single Video Inference**: ~2-5 seconds (GPU) / ~30 seconds (CPU)
- **Batch Inference**: Processes multiple videos in parallel
- **Memory Usage**: ~4 GB GPU VRAM for batch size 32
- **Training Time**: ~24 hours for 50 epochs on RTX 4090

---

**Last Updated**: March 2025
