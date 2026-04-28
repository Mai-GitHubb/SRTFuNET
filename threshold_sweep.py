"""
threshold_sweep.py (Ensemble-Aware Version with TTA)

Auto-calibrates temperature and sweeps thresholds for a single checkpoint 
OR an ensemble of top-k checkpoints.
Supports Test-Time Augmentation (TTA) for improved robustness.
"""

import os, sys, argparse, glob

# -- Warning suppression --
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['GLOG_minloglevel']      = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from late_fusion_model import LateFusionDeepfakeDetector
from dataset import LateFusionDataset

def _worker_init(worker_id):
    import os, sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.stderr = open(os.devnull, 'w')

def load_checkpoint(path, model, device):
    try:
        import numpy._core.multiarray as _npcma
        torch.serialization.add_safe_globals([_npcma.scalar])
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)

def build_model(path, device):
    model = LateFusionDeepfakeDetector()
    ckpt = load_checkpoint(path, model, device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def calibrate(self, logits, labels):
        self.to('cpu')
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)
        criterion = nn.CrossEntropyLoss()
        def step():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature.clamp(min=1e-3), labels)
            loss.backward()
            return loss
        optimizer.step(step)
        return self.temperature.item()

class TTADatasetWrapper(Dataset):
    """Wrapper that applies Test-Time Augmentation to a dataset."""
    def __init__(self, base_dataset, num_augmentations=4):
        """
        Args:
            base_dataset: LateFusionDataset instance
            num_augmentations: Number of augmented versions per sample (0 = no TTA)
        """
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        batch = self.base_dataset[idx]
        
        if self.num_augmentations == 0:
            # No TTA: return single sample
            return batch
        
        # Apply TTA augmentations to spatial stream
        spatial = batch['spatial']
        tta_samples = [spatial]  # Include original
        
        for _ in range(self.num_augmentations):
            augmented = spatial.clone()
            
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                augmented = TF.hflip(augmented)
            
            # Random color jitter
            if torch.rand(1).item() < 0.5:
                jitter = T.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.05
                )
                augmented = jitter(augmented)
            
            # Random rotation (small)
            if torch.rand(1).item() < 0.5:
                angle = torch.randint(-15, 16, (1,)).item()
                augmented = TF.rotate(augmented, angle)
            
            # Random Gaussian noise
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(augmented) * 0.05
                augmented = torch.clamp(augmented + noise, 0.0, 1.0)
            
            tta_samples.append(augmented)
        
        # Stack augmented versions
        batch['spatial_tta'] = torch.stack(tta_samples)  # (num_aug+1, C, H, W)
        # Also keep original for temporal and landmark (no TTA for those)
        batch['temporal_tta'] = batch['temporal'].unsqueeze(0).expand(
            len(tta_samples), -1, -1, -1, -1
        )  # (num_aug+1, T, C, H, W)
        batch['landmark_tta'] = batch['landmark'].unsqueeze(0).expand(
            len(tta_samples), -1
        )  # (num_aug+1, landmark_dim)
        
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_auc.pth')
    parser.add_argument('--ensemble', action='store_true', help='Sweep across ensemble of top-k models')
    parser.add_argument('--tta', type=int, default=0, help='Number of TTA augmentations (0 = no TTA)')
    args = parser.parse_args()

    TEST_LIST_PATH  = r"D:\Deepfake\SRTfuNET\data\List_of_testing_videos.txt"
    DATA_ROOT       = r"D:\Deepfake\SRTfuNET\data\Test_videos"
    BATCH_SIZE      = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Gather Models
    models = []
    if args.ensemble:
        paths = sorted(glob.glob('checkpoints/topk_*.pth'))
        if not paths: paths = ['checkpoints/best_auc.pth', 'checkpoints/best_balanced.pth']
        print(f"Ensembling {len(paths)} models for sweep...")
        for p in paths:
            if os.path.exists(p): models.append(build_model(p, device))
    else:
        print(f"Single model sweep: {args.checkpoint}")
        models.append(build_model(args.checkpoint, device))

    # 2. Load Data (with updated path)
    test_videos, test_labels = [], []
    with open(TEST_LIST_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Update path to point to Test_videos subfolder
                video_path = parts[1].replace('/', os.sep)
                test_videos.append(os.path.join(DATA_ROOT, video_path))
                test_labels.append(1 - int(parts[0]))

    base_dataset = LateFusionDataset(test_videos, test_labels, is_training=False)
    
    # Wrap with TTA if requested
    if args.tta > 0:
        print(f"Using Test-Time Augmentation with {args.tta} augmentations")
        dataset = TTADatasetWrapper(base_dataset, num_augmentations=args.tta)
    else:
        dataset = base_dataset
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE, num_workers=0, worker_init_fn=None if args.tta > 0 else _worker_init
    )

    # 3. Collect Ensemble Logits (with TTA support)
    all_logits, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting Ensemble Logits"):
            if args.tta > 0:
                # TTA mode: batch contains stacked augmentations
                s = batch['spatial_tta'].to(device)  # (batch, num_aug+1, C, H, W)
                t = batch['temporal_tta'].to(device)  # (batch, num_aug+1, T, C, H, W)
                l = batch['landmark_tta'].to(device)  # (batch, num_aug+1, landmark_dim)
                
                batch_size = s.size(0)
                num_aug = s.size(1)
                
                # Process each augmentation and average
                batch_logits = torch.zeros((batch_size, 2), device=device)
                
                for aug_idx in range(num_aug):
                    s_aug = s[:, aug_idx]  # (batch, C, H, W)
                    t_aug = t[:, aug_idx]  # (batch, T, C, H, W)
                    l_aug = l[:, aug_idx]  # (batch, landmark_dim)
                    
                    aug_logits = torch.zeros((batch_size, 2), device=device)
                    for m in models:
                        aug_logits += m(s_aug, t_aug, l_aug)
                    aug_logits /= len(models)
                    
                    batch_logits += aug_logits
                
                # Average across augmentations
                batch_logits /= num_aug
            else:
                # Standard mode
                s, t, l = batch['spatial'].to(device), batch['temporal'].to(device), batch['landmark'].to(device)
                
                batch_logits = torch.zeros((s.size(0), 2), device=device)
                for m in models:
                    batch_logits += m(s, t, l)
                batch_logits /= len(models)
            
            all_logits.append(batch_logits.cpu())
            all_true.extend(batch['label'].numpy())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.tensor(all_true, dtype=torch.long)
    true_arr   = np.array(all_true)

    # 4. Calibrate T
    scaler = TemperatureScaler()
    T = scaler.calibrate(logits_cat, labels_cat)
    probs_arr = torch.softmax(logits_cat / max(T, 1e-3), dim=1)[:, 1].numpy()

    # 5. Sweep Thresholds
    thresholds = np.arange(0.30, 0.95, 0.01)
    best_bal_thr, best_bal_val = 0.5, 0.0
    
    print(f"\n{'Thresh':>7} | {'Overall':>8} | {'Real':>8} | {'Fake':>8} | {'Min(R,F)':>9}")
    print("-" * 55)

    for thr in thresholds:
        preds = (probs_arr >= thr).astype(int)
        real_acc = (preds[true_arr==0] == 0).mean()
        fake_acc = (preds[true_arr==1] == 1).mean()
        balanced = min(real_acc, fake_acc)
        
        if balanced > best_bal_val:
            best_bal_val, best_bal_thr = balanced, thr
            
        if int(thr*100) % 5 == 0: # Print every 0.05 for brevity
            print(f"{thr:7.2f} | {accuracy_score(true_arr, preds):8.4f} | {real_acc:8.4f} | {fake_acc:8.4f} | {balanced:9.4f}")

    print(f"\nENSEMBLE RESULTS{'(with TTA)' if args.tta > 0 else ''}:")
    print(f"Optimal Temperature : {T:.4f}")
    print(f"Balanced Threshold  : {best_bal_thr:.2f}")
    print(f"Min Class Accuracy  : {best_bal_val:.4f}")
    if args.tta > 0:
        print(f"TTA Augmentations   : {args.tta}")

if __name__ == '__main__':
    main()