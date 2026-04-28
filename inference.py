# ── Warning suppression ───────────────────────────────────────────────────────
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['GLOG_minloglevel']      = '3'
os.environ['GLOG_logtostderr']      = '0'
os.environ['TF_LITE_LOG_SEVERITY']  = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['ABSL_MIN_LOG_LEVEL']    = '3'

import warnings
warnings.filterwarnings("ignore")

try:
    import absl.logging as _absl_log
    _absl_log.set_verbosity(_absl_log.ERROR)
    _absl_log.use_absl_handler()
except ImportError:
    pass

def _silence_fd2():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved

def _restore_fd2(saved):
    os.dup2(saved, 2)
    os.close(saved)

_s = _silence_fd2()
try:
    import mediapipe as _mp
finally:
    _restore_fd2(_s)


def _worker_init(worker_id):
    import os, sys, warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
    os.environ['GLOG_minloglevel']      = '3'
    os.environ['GLOG_logtostderr']      = '0'
    os.environ['TF_LITE_LOG_SEVERITY']  = '3'
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['ABSL_MIN_LOG_LEVEL']    = '3'
    warnings.filterwarnings("ignore")
    try:
        import absl.logging as _al
        _al.set_verbosity(_al.ERROR)
    except ImportError:
        pass
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        sys.stderr = open(os.devnull, 'w')
    except OSError:
        pass


warnings.filterwarnings("ignore")

import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# Add these to your imports in inf.py
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from late_fusion_model import LateFusionDeepfakeDetector
from dataset import LateFusionDataset


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path, model, device):
    try:
        import numpy._core.multiarray as _npcma
        torch.serialization.add_safe_globals([_npcma.scalar])
        ckpt = torch.load(path, map_location=device, weights_only=True)
        print(f"  Loaded {os.path.basename(path)} (weights_only=True)")
        return ckpt
    except Exception:
        pass
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"  Loaded {os.path.basename(path)} (weights_only=False fallback)")
    return ckpt


def build_model_from_ckpt(path, device):
    """Instantiate a fresh model and load weights from checkpoint."""
    model = LateFusionDeepfakeDetector()
    ckpt  = load_checkpoint(path, model, device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        sep = ckpt.get('separation', None)
        print(f"    Epoch {ckpt.get('epoch','?')} | "
              f"Val AUC: {ckpt.get('val_auc',0):.4f} | "
              f"Val Real: {ckpt.get('real_acc',0):.4f} | "
              f"Val Fake: {ckpt.get('fake_acc',0):.4f}" +
              (f" | Sep: {sep:.4f}" if sep is not None else ""))
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_tta(model, batch, device, temperature, n_tta=3):
    """
    Average softmax probabilities over n_tta augmented views.
    """
    spatial  = batch['spatial' ].to(device)
    temporal = batch['temporal'].to(device)
    landmark = batch['landmark'].to(device)

    accumulated = torch.zeros(spatial.size(0), device=device)

    with torch.no_grad():
        for i in range(n_tta):
            s = spatial
            if i == 1:
                # Flip left–right along the width dimension (dim=-1)
                s = torch.flip(spatial, dims=[-1])
            logits = model(s, temporal, landmark)
            scaled = logits / max(temperature, 1e-3)
            probs  = torch.softmax(scaled, dim=1)[:, 1]
            accumulated += probs

    return accumulated / n_tta


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble inference
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(models, loader, device, temperature, n_tta, desc):
    """
    Run inference with a list of models and average their TTA probabilities.
    """
    all_probs, all_true = [], []

    for batch in tqdm(loader, desc=desc, file=sys.stdout):
        batch_probs = torch.zeros(batch['spatial'].size(0), device=device)

        for model in models:
            batch_probs += predict_with_tta(model, batch, device, temperature, n_tta)

        batch_probs /= len(models)
        all_probs.extend(batch_probs.cpu().numpy())
        all_true.extend(batch['label'].numpy())

    return np.array(all_probs), np.array(all_true)

def plot_visualizations(probs_arr, true_arr, threshold, auc_score):
    """
    Generates and saves visual plots for model evaluation.
    """
    preds_arr = (probs_arr >= threshold).astype(int)
    os.makedirs('results', exist_ok=True)
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── 1. Confusion Matrix ──
    cm = confusion_matrix(true_arr, preds_arr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[0].set_title(f'Confusion Matrix (Thr={threshold})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # ── 2. ROC Curve ──
    fpr, tpr, _ = roc_curve(true_arr, probs_arr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc="lower right")

    # ── 3. Probability Distributions ──
    real_probs = probs_arr[true_arr == 0]
    fake_probs = probs_arr[true_arr == 1]
    sns.kdeplot(real_probs, fill=True, color="blue", label="Real", ax=axes[2])
    sns.kdeplot(fake_probs, fill=True, color="red", label="Fake", ax=axes[2])
    axes[2].axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    axes[2].set_title('Probability Separation')
    axes[2].set_xlabel('Predicted Probability of "Fake"')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('results/inference_metrics.png')
    print("\n[Visuals] Metrics plots saved to: results/inference_metrics.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Results printing
# ─────────────────────────────────────────────────────────────────────────────

def print_results(probs_arr, true_arr, temperature, threshold, n_videos):
    preds_arr = (probs_arr >= threshold).astype(int)
    real_mask = true_arr == 0
    fake_mask = true_arr == 1

    # Standard Metrics
    acc = accuracy_score(true_arr, preds_arr)
    auc = roc_auc_score(true_arr, probs_arr)
    prec = precision_score(true_arr, preds_arr)
    rec = recall_score(true_arr, preds_arr)
    f1 = f1_score(true_arr, preds_arr)
    
    # Confusion Matrix
    cm = confusion_matrix(true_arr, preds_arr)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + " TEST SET PERFORMANCE ".center(60, "═"))
    print(f"  Temperature   : {temperature:.4f} | Threshold: {threshold:.2f}")
    print(f"  Total Videos  : {n_videos}")
    print("-" * 60)
    print(f"  AUC           : {auc:.4f} ⭐")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Precision     : {prec:.4f} (Ability to avoid false alarms)")
    print(f"  Recall        : {rec:.4f} (Ability to find all fakes)")
    print(f"  F1-Score      : {f1:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"                Predicted Real    Predicted Fake")
    print(f"  Actual Real |      {tn:<10}       {fp:<10}  (TN / FP)")
    print(f"  Actual Fake |      {fn:<10}       {tp:<10}  (FN / TP)")
    print("-" * 60)

    real_acc = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fake_acc = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    print(f"  Real Accuracy : {real_acc:.4f}")
    print(f"  Fake Accuracy : {fake_acc:.4f}")
    
    sep = float(np.median(probs_arr[fake_mask]) - np.median(probs_arr[real_mask]))
    print(f"  Median Sep    : {sep:.4f} {'(Strong)' if sep > 0.5 else '(Moderate)'}")
    print("═" * 60)

    # Detailed report for precision/recall per class
    print("\nDetailed Classification Report:")
    print(classification_report(true_arr, preds_arr, target_names=['Real', 'Fake']))
    plot_visualizations(probs_arr, true_arr, threshold, auc)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with optional TTA and ensemble."
    )
    parser.add_argument(
        '--ensemble', action='store_true',
        help='Load all top-k checkpoints from checkpoints/topk_*.pth and ensemble them.'
    )
    parser.add_argument(
        '--tta', type=int, default=3, metavar='N',
        help='Number of TTA passes per model (default: 3). Set to 1 to disable TTA.'
    )
    # --- ADDED ARGUMENTS FOR AUTO_EVAL SCRIPT ---
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints/best_auc.pth',
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Temperature scaling value'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold'
    )
    args = parser.parse_args()

    BATCH_SIZE      = 8
    TEST_LIST_PATH  = r"D:\Deepfake\SRTfuNET\data\List_of_testing_videos.txt"
    DATA_ROOT       = r"D:\Deepfake\SRTfuNET\data\Test_videos"
    NUM_FRAMES      = 16

    # --- USING THE ARGUMENTS ---
    CHECKPOINT_PATH = args.checkpoint
    TEMPERATURE     = args.temperature
    THRESHOLD       = args.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tta  = max(1, args.tta)

    print(f"Using device: {device}  |  T={TEMPERATURE}  |  Threshold={THRESHOLD}")
    print(f"TTA passes   : {n_tta}")
    print(f"Ensemble mode: {'yes' if args.ensemble else 'no'}\n")

    # ── Test list ─────────────────────────────────────────────────────────────
    print("Loading test dataset paths...")
    test_videos, test_labels = [], []
    with open(TEST_LIST_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            internal_label = 1 - int(parts[0])
            path_full = os.path.join(DATA_ROOT, parts[1].replace('/', os.sep))
            if os.path.exists(path_full):
                test_videos.append(path_full)
                test_labels.append(internal_label)
            else:
                print(f"  Warning: not found — {path_full}")

    print(f"Found {len(test_videos)} test videos "
          f"(real: {test_labels.count(0)}, fake: {test_labels.count(1)})")

    # ── Dataset / loader ──────────────────────────────────────────────────────
    test_dataset = LateFusionDataset(
        video_paths=test_videos, labels=test_labels,
        is_training=False, num_frames=NUM_FRAMES,
    )
    # Disable num_workers when TTA or ensemble is enabled to avoid memory issues
    # (ensemble with 3+ models is memory-intensive)
    num_workers = 0 if (n_tta > 1 or args.ensemble) else 4
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )

    # ── Load model(s) ─────────────────────────────────────────────────────────
    if args.ensemble:
        # Load all top-k checkpoints
        topk_paths = sorted(glob.glob('checkpoints/topk_*.pth'))
        if not topk_paths:
            print("No top-k checkpoints found. Falling back to best_auc.pth and best_balanced.pth.")
            topk_paths = [p for p in ['checkpoints/best_auc.pth',
                                       'checkpoints/best_balanced.pth']
                          if os.path.exists(p)]
        if not topk_paths:
            raise FileNotFoundError("No checkpoints found for ensemble.")

        print(f"\nEnsemble: loading {len(topk_paths)} checkpoints...")
        models = []
        for path in topk_paths:
            print(f"  {path}")
            m = build_model_from_ckpt(path, device)
            models.append(m)

        print(f"\nRunning ensemble inference (TTA={n_tta} per model)...")
        probs_arr, true_arr = run_ensemble(
            models, test_loader, device, TEMPERATURE, n_tta,
            desc=f"[Ensemble x{len(models)}, TTA x{n_tta}]"
        )

    else:
        # Single checkpoint
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
        model = build_model_from_ckpt(CHECKPOINT_PATH, device)

        print(f"\nRunning inference (TTA={n_tta})...")
        all_probs, all_true = [], []
        for batch in tqdm(test_loader, desc=f"[Inference, TTA x{n_tta}]",
                          file=sys.stdout):
            probs = predict_with_tta(model, batch, device, TEMPERATURE, n_tta)
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(batch['label'].numpy())

        probs_arr = np.array(all_probs)
        true_arr  = np.array(all_true)

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(probs_arr, true_arr, TEMPERATURE, THRESHOLD, len(test_videos))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()