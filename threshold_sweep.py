"""
threshold_sweep.py (Ensemble-Aware Version)

Auto-calibrates temperature and sweeps thresholds for a single checkpoint 
OR an ensemble of top-k checkpoints.
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_auc.pth')
    parser.add_argument('--ensemble', action='store_true', help='Sweep across ensemble of top-k models')
    args = parser.parse_args()

    TEST_LIST_PATH  = r"D:\Deepfake\SRTfuNET\data\List_of_testing_videos.txt"
    DATA_ROOT       = r"D:\Deepfake\SRTfuNET\data"
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

    # 2. Load Data
    test_videos, test_labels = [], []
    with open(TEST_LIST_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                test_videos.append(os.path.join(DATA_ROOT, parts[1].replace('/', os.sep)))
                test_labels.append(1 - int(parts[0]))

    loader = DataLoader(
        LateFusionDataset(test_videos, test_labels, is_training=False),
        batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=_worker_init
    )

    # 3. Collect Ensemble Logits
    all_logits, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting Ensemble Logits"):
            s, t, l = batch['spatial'].to(device), batch['temporal'].to(device), batch['landmark'].to(device)
            
            # Average logits across all models
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

    print(f"\nENSEMBLE RESULTS:")
    print(f"Optimal Temperature : {T:.4f}")
    print(f"Balanced Threshold  : {best_bal_thr:.2f}")
    print(f"Min Class Accuracy  : {best_bal_val:.4f}")

if __name__ == '__main__':
    main()