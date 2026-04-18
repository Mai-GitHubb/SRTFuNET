import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                             confusion_matrix, precision_score, recall_score, f1_score)
from tqdm import tqdm

from late_fusion_model import LateFusionDeepfakeDetector
from dataset import LateFusionDataset

# ── Calibrated Config ─────────────────────────────────────────────────────────
TEMPERATURE = 1.3630  
THRESHOLD   = 0.4600  
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_LIST   = r"D:\Deepfake\SRTfuNET\data\List_of_testing_videos.txt"
DATA_ROOT   = r"D:\Deepfake\SRTfuNET\data"
BATCH_SIZE  = 8
USE_ENSEMBLE = True
N_TTA       = 1  # Test-Time Augmentation passes (1 = no TTA)

def load_checkpoint(path, model, device):
    """Load checkpoint with weights_only fallback."""
    try:
        import numpy._core.multiarray as _npcma
        torch.serialization.add_safe_globals([_npcma.scalar])
        ckpt = torch.load(path, map_location=device, weights_only=True)
        return ckpt
    except Exception:
        pass
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def build_model_from_ckpt(path, device):
    """Instantiate a fresh model and load weights from checkpoint."""
    model = LateFusionDeepfakeDetector()
    ckpt  = load_checkpoint(path, model, device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def load_ensemble_models(device):
    """Load all top-k checkpoints for ensemble."""
    topk_paths = sorted(glob.glob('checkpoints/topk_*.pth'))
    if not topk_paths:
        print("No top-k checkpoints found. Using single checkpoint fallback...")
        topk_paths = [p for p in ['checkpoints/best_auc.pth', 'checkpoints/best_balanced.pth']
                      if os.path.exists(p)]
    
    if not topk_paths:
        raise FileNotFoundError("No checkpoints found.")
    
    print(f"Loading {len(topk_paths)} checkpoints for ensemble...")
    models = []
    for path in topk_paths:
        print(f"  {path}")
        m = build_model_from_ckpt(path, device)
        models.append(m)
    
    return models


def predict_with_tta(model, s, t, l, device, temperature, n_tta=1):
    """
    Average softmax probabilities over n_tta augmented views (TTA).
    Applies horizontal flips as augmentation.
    """
    accumulated = torch.zeros(s.size(0), device=device)

    with torch.no_grad():
        for i in range(n_tta):
            s_aug = s
            if i == 1:
                # Flip left-right along the width dimension (dim=-1)
                s_aug = torch.flip(s, dims=[-1])
            
            logits = model(s_aug, t, l)
            scaled = logits / max(temperature, 1e-3)
            probs = torch.softmax(scaled, dim=1)[:, 1]
            accumulated += probs

    return accumulated / n_tta


def get_metrics_and_plots(probs, labels, mode):
    preds = (probs >= THRESHOLD).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {mode}')
    
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, color='orange', label=f'AUC={auc:.4f}')
    axes[1].plot([0, 1], [0, 1], linestyle='--')
    axes[1].set_title(f'ROC Curve: {mode}')
    axes[1].legend()

    sns.kdeplot(probs[labels==0], fill=True, label="Real", ax=axes[2])
    sns.kdeplot(probs[labels==1], fill=True, label="Fake", ax=axes[2])
    axes[2].set_title(f'Probability Separation: {mode}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'results/ablation_{mode}.png')
    plt.close()
    
    return [auc, acc, p, r, f1]

def evaluate_stream(models, loader, stream_type):
    """Evaluate ensemble on a specific stream (masking others) with TTA."""
    all_probs, all_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {stream_type}"):
            s = batch['spatial'].to(DEVICE)
            t = batch['temporal'].to(DEVICE)
            l = batch['landmark'].to(DEVICE)

            # Masking: Zero out the streams we AREN'T testing
            if stream_type == 'spatial':
                t, l = torch.zeros_like(t), torch.zeros_like(l)
            elif stream_type == 'temporal':
                s, l = torch.zeros_like(s), torch.zeros_like(l)
            elif stream_type == 'landmark':
                s, t = torch.zeros_like(s), torch.zeros_like(t)

            # Ensemble + TTA: average predictions from all models with TTA
            batch_probs = torch.zeros(s.size(0), device=DEVICE)
            for model in models:
                probs = predict_with_tta(model, s, t, l, DEVICE, TEMPERATURE, N_TTA)
                batch_probs += probs
            
            batch_probs /= len(models)
            all_probs.extend(batch_probs.cpu().numpy())
            all_true.extend(batch['label'].numpy())
    
    return np.array(all_probs), np.array(all_true)

def main():
    os.makedirs('results', exist_ok=True)
    
    # Load ensemble models
    if USE_ENSEMBLE:
        models = load_ensemble_models(DEVICE)
        print(f"✓ Loaded {len(models)} models for ensemble evaluation\n")
    else:
        single_model = LateFusionDeepfakeDetector().to(DEVICE)
        ckpt = torch.load('checkpoints/topk_e07_auc0.9736.pth', map_location=DEVICE)
        single_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        models = [single_model]
        print(f"✓ Loaded single model\n")
    
    # Load test data
    videos, labels = [], []
    with open(TEST_LIST) as f:
        for line in f:
            parts = line.strip().split()
            videos.append(os.path.join(DATA_ROOT, parts[1].replace('/', os.sep)))
            labels.append(1 - int(parts[0]))
    
    loader = DataLoader(LateFusionDataset(videos, labels, is_training=False), batch_size=BATCH_SIZE)

    # Evaluate each stream
    print(f"Temperature: {TEMPERATURE} | Threshold: {THRESHOLD} | TTA: {N_TTA}\n")
    results = {}
    for mode in ['spatial', 'temporal', 'landmark']:
        probs, true = evaluate_stream(models, loader, mode)
        results[mode] = get_metrics_and_plots(probs, true, mode)

    # Print summary
    print("\n" + "="*70)
    print(f"{'Stream':<12} | {'AUC':<7} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 70)
    for k, v in results.items():
        print(f"{k:<12} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | {v[4]:.4f}")
    print("="*70)
    print("\n✓ Ablation evaluation complete. Plots saved to results/ablation_*.png")


if __name__ == "__main__":
    main()