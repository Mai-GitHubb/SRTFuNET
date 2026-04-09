# ── Warning suppression — MUST be first ──────────────────────────────────────
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

_saved_fd2 = _silence_fd2()
try:
    import mediapipe as _mp
finally:
    _restore_fd2(_saved_fd2)


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

import heapq
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from late_fusion_model import LateFusionDeepfakeDetector
from dataset import LateFusionDataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.05, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        device      = inputs.device
        with torch.no_grad():
            smooth_targets = torch.full(
                inputs.shape, self.smoothing / (num_classes - 1), device=device,
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs  = torch.nn.functional.log_softmax(inputs, dim=1)
        ce_loss    = -(smooth_targets * log_probs).sum(dim=1)
        with torch.no_grad():
            probs = torch.softmax(inputs, dim=1)
            pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha.to(device)[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def mixup_collate(batch, alpha=0.2):
    spatial  = torch.stack([b['spatial']  for b in batch])
    temporal = torch.stack([b['temporal'] for b in batch])
    landmark = torch.stack([b['landmark'] for b in batch])
    labels   = torch.tensor([b['label']   for b in batch], dtype=torch.long)
    if alpha > 0 and torch.rand(1).item() < 0.5:
        lam   = float(np.random.beta(alpha, alpha))
        lam   = max(lam, 1.0 - lam)
        index = torch.randperm(len(batch))
        spatial  = lam * spatial  + (1.0 - lam) * spatial[index]
        temporal = lam * temporal + (1.0 - lam) * temporal[index]
        landmark = lam * landmark + (1.0 - lam) * landmark[index]
        labels   = torch.where(torch.tensor(lam >= 0.5), labels, labels[index])
    return {'spatial': spatial, 'temporal': temporal,
            'landmark': landmark, 'label': labels}


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    # These are the exact settings from the run that produced:
    #   val AUC  = 0.9349  (epoch 19)
    #   test AUC = 0.838   (with threshold_sweep + inference.py)
    #
    # Additions vs that run:
    #   - dual checkpoint saving (best_auc.pth + best_balanced.pth)
    #   - top-3 epoch checkpoints
    #   - dataset.py now uses 1405-d landmarks (detection flag)
    EPOCHS          = 50
    BATCH_SIZE      = 8
    MAX_LR          = 5e-5
    GRAD_CLIP       = 1.0
    NUM_FRAMES      = 16
    MIXUP_ALPHA     = 0.2
    LABEL_SMOOTHING = 0.05
    FOCAL_ALPHA     = 1.5
    FOCAL_GAMMA     = 2.0
    PATIENCE        = 10
    WEIGHT_DECAY    = 1e-4
    TOP_K           = 3
    CHECKPOINT_DIR  = 'checkpoints'
    CKPT_AUC        = os.path.join(CHECKPOINT_DIR, 'best_auc.pth')
    CKPT_BALANCED   = os.path.join(CHECKPOINT_DIR, 'best_balanced.pth')

    DATA_DIRS = [
        r"D:\Deepfake\SRTfuNET\data\Celeb-synthesis",
        r"D:\Deepfake\SRTfuNET\data\Celeb-real",
        r"D:\Deepfake\SRTfuNET\data\YouTube-real",
    ]
    EXCLUSION_FILE = r"D:\Deepfake\SRTfuNET\data\List_of_testing_videos.txt"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build video list ──────────────────────────────────────────────────────
    with open(EXCLUSION_FILE) as f:
        excluded = {
            line.strip().split()[-1].replace('\\', '/')
            for line in f if line.strip()
        }
    print(f"Excluding {len(excluded)} test videos.")

    all_videos, all_labels = [], []
    for data_dir in DATA_DIRS:
        label = 1 if 'synthesis' in os.path.basename(data_dir) else 0
        for root, _, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.mp4'):
                    continue
                rel = f"{os.path.basename(root)}/{file}"
                if rel not in excluded:
                    all_videos.append(os.path.join(root, file))
                    all_labels.append(label)

    count_0 = all_labels.count(0)
    count_1 = all_labels.count(1)
    print(f"Total: {len(all_videos)} | Real: {count_0} | Fake: {count_1} | "
          f"Ratio: {count_1/max(count_0,1):.1f}:1")

    # ── Train / val split ─────────────────────────────────────────────────────
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        all_videos, all_labels,
        test_size=0.2, random_state=42, stratify=all_labels,
    )
    tr_r = train_labels.count(0)
    tr_f = train_labels.count(1)
    print(f"Train: {len(train_videos)} (real={tr_r}, fake={tr_f}) | "
          f"Val: {len(val_videos)}\n")

    train_dataset = LateFusionDataset(
        video_paths=train_videos, labels=train_labels,
        is_training=True,  num_frames=NUM_FRAMES,
    )
    val_dataset = LateFusionDataset(
        video_paths=val_videos, labels=val_labels,
        is_training=False, num_frames=NUM_FRAMES,
    )

    # WeightedRandomSampler balances 7.4:1 imbalance to ~1:1 per batch
    sample_weights = [
        1.0 / tr_r if lbl == 0 else 1.0 / tr_f
        for lbl in train_labels
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init,
        collate_fn=partial(mixup_collate, alpha=MIXUP_ALPHA),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        worker_init_fn=_worker_init,
    )
    print("DataLoaders ready.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model            = LateFusionDeepfakeDetector().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        trainable_params, lr=MAX_LR / 25, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR,
        steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=25, final_div_factor=1e4,
    )

    alpha     = torch.tensor([FOCAL_ALPHA, 1.0], dtype=torch.float32)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha, smoothing=LABEL_SMOOTHING)
    print(f"FocalLoss — alpha[real]={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}, "
          f"smoothing={LABEL_SMOOTHING}")
    print(f"Optimiser — MAX_LR={MAX_LR}, weight_decay={WEIGHT_DECAY}, "
          f"patience={PATIENCE}")
    print(f"Saving: {CKPT_AUC} + {CKPT_BALANCED} + top-{TOP_K}\n")

    best_val_auc  = 0.0
    best_balanced = 0.0
    patience_count = 0
    top_k_heap    = []

    for epoch in range(EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss            = 0.0
        train_preds, train_true = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]",
                    file=sys.stdout)
        for batch in pbar:
            spatial  = batch['spatial' ].to(device)
            temporal = batch['temporal'].to(device)
            landmark = batch['landmark'].to(device)
            labels   = batch['label'   ].to(device)

            optimizer.zero_grad()
            outputs = model(spatial, temporal, landmark)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        tr_arr     = np.array(train_true)
        tr_pred    = np.array(train_preds)
        train_loss = running_loss / len(train_loader)
        train_acc  = accuracy_score(tr_arr, tr_pred)
        print(f"\nEpoch {epoch+1:02d} | Train loss: {train_loss:.4f} | "
              f"Acc: {train_acc:.4f} | "
              f"Real: {float((tr_pred[tr_arr==0]==0).mean()) if (tr_arr==0).any() else 0:.4f} | "
              f"Fake: {float((tr_pred[tr_arr==1]==1).mean()) if (tr_arr==1).any() else 0:.4f}")

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        running_loss               = 0.0
        val_preds, val_probs, val_true = [], [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  ",
                        file=sys.stdout)
            for batch in pbar:
                spatial  = batch['spatial' ].to(device)
                temporal = batch['temporal'].to(device)
                landmark = batch['landmark'].to(device)
                labels   = batch['label'   ].to(device)

                outputs      = model(spatial, temporal, landmark)
                loss         = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_loss  = running_loss / len(val_loader)
        val_acc   = accuracy_score(val_true, val_preds)
        val_auc   = roc_auc_score(val_true, val_probs)
        val_arr   = np.array(val_true)
        prob_arr  = np.array(val_probs)
        pred_arr  = np.array(val_preds)
        real_acc  = float((pred_arr[val_arr==0]==0).mean()) if (val_arr==0).any() else 0.0
        fake_acc  = float((pred_arr[val_arr==1]==1).mean()) if (val_arr==1).any() else 0.0
        real_med  = float(np.median(prob_arr[val_arr==0])) if (val_arr==0).any() else 0.5
        fake_med  = float(np.median(prob_arr[val_arr==1])) if (val_arr==1).any() else 0.5
        sep       = fake_med - real_med
        gap       = train_acc - val_acc
        balanced_score = min(real_acc, fake_acc) if (real_acc >= 0.60 and fake_acc >= 0.60) else 0.0

        print(f"           Val  loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"AUC: {val_auc:.4f} | Real: {real_acc:.4f} | Fake: {fake_acc:.4f}")
        print(f"           Sep: {sep:.4f}  (real={real_med:.3f}, fake={fake_med:.3f})  "
              f"Gap: {gap:.4f}  {'⚠ overfit' if gap > 0.12 else '✓ generalising'}")
        print(f"           Sep: {'✓ strong' if sep > 0.45 else '~ moderate' if sep > 0.30 else '⚠ low'}  "
              f"| Balanced: {balanced_score:.4f}")

        if real_acc > 0.65 and fake_acc > 0.65:
            print("  ✓ Both classes balanced.")
        elif fake_acc > 0.90 and real_acc < 0.40:
            print("  ⚠ Biased toward fake.")

        ckpt_data = {
            'epoch'           : int(epoch + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'val_auc'         : float(val_auc),
            'val_acc'         : float(val_acc),
            'real_acc'        : float(real_acc),
            'fake_acc'        : float(fake_acc),
            'separation'      : float(sep),
            'real_median'     : float(real_med),
            'fake_median'     : float(fake_med),
            'train_val_gap'   : float(gap),
            'balanced_score'  : float(balanced_score),
        }

        saved_auc = saved_bal = False

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ckpt_data, CKPT_AUC)
            saved_auc = True

        if balanced_score > best_balanced:
            best_balanced = balanced_score
            torch.save(ckpt_data, CKPT_BALANCED)
            saved_bal = True

        # ── Top-K ─────────────────────────────────────────────────────────────
        ckpt_k_path = os.path.join(
            CHECKPOINT_DIR, f'topk_e{epoch+1:02d}_auc{val_auc:.4f}.pth'
        )
        if len(top_k_heap) < TOP_K:
            torch.save(ckpt_data, ckpt_k_path)
            heapq.heappush(top_k_heap, (val_auc, epoch, ckpt_k_path))
        elif val_auc > top_k_heap[0][0]:
            _, _, old_path = heapq.heappop(top_k_heap)
            if os.path.exists(old_path):
                os.remove(old_path)
            torch.save(ckpt_data, ckpt_k_path)
            heapq.heappush(top_k_heap, (val_auc, epoch, ckpt_k_path))

        if saved_auc and saved_bal:
            print(f"  ✓ Both checkpoints updated — AUC: {val_auc:.4f} | "
                  f"Balanced: {balanced_score:.4f} (Real: {real_acc:.4f}, Fake: {fake_acc:.4f})")
        elif saved_auc:
            print(f"  ✓ best_auc.pth updated — AUC: {val_auc:.4f}")
        elif saved_bal:
            print(f"  ✓ best_balanced.pth updated — Balanced: {balanced_score:.4f} "
                  f"(Real: {real_acc:.4f}, Fake: {fake_acc:.4f})")

        if not saved_auc:
            patience_count += 1
            print(f"  No AUC improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break
        else:
            patience_count = 0

    print(f"\nTraining complete.")
    print(f"  best_auc.pth      → AUC: {best_val_auc:.4f}")
    print(f"  best_balanced.pth → Balanced: {best_balanced:.4f}")
    print(f"\nNext steps:")
    print(f"  python threshold_sweep.py --checkpoint checkpoints/best_auc.pth")
    print(f"  python threshold_sweep.py --checkpoint checkpoints/best_balanced.pth")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()