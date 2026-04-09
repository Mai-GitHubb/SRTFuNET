import os
import glob
import subprocess
import re

CHECKPOINTS_DIR = "checkpoints"

def main():
    # Find all .pth files in the checkpoints folder
    pth_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pth"))
    
    if not pth_files:
        print("No checkpoints found in the 'checkpoints' directory.")
        return

    results = []

    print(f"Found {len(pth_files)} checkpoints to evaluate...\n")

    for ckpt in pth_files:
        ckpt_name = os.path.basename(ckpt)
        print(f"{'='*70}")
        print(f"🚀 Evaluating: {ckpt_name}")
        print(f"{'='*70}")

        # 1. Run threshold_sweep.py
        print("1️⃣ Running threshold sweep & temperature calibration...")
        sweep_cmd = ["python", "threshold_sweep.py", "--checkpoint", ckpt]
        sweep_result = subprocess.run(sweep_cmd, capture_output=True, text=True)

        # Parse Temperature and Threshold from the sweep output
        t_match = re.search(r"Calibrated T\s*:\s*([0-9.]+)", sweep_result.stdout)
        thr_match = re.search(r"Balanced thr\s*:\s*([0-9.]+)", sweep_result.stdout)

        if not t_match or not thr_match:
            print(f"❌ Failed to parse temperature or threshold for {ckpt_name}. Skipping.")
            continue

        temperature = float(t_match.group(1))
        threshold = float(thr_match.group(1))
        print(f"   -> Calibrated Temp: {temperature:.4f} | Optimal Threshold: {threshold:.2f}")

        # 2. Run inference.py with the parsed arguments
        print("2️⃣ Running Deepfake inference on Test Set...")
        inf_cmd = [
            "python", "inf.py",
            "--checkpoint", ckpt,
            "--temperature", str(temperature),
            "--threshold", str(threshold),
            "--tta", "3"  # You can change this to 1 if you want it to evaluate faster
        ]
        inf_result = subprocess.run(inf_cmd, capture_output=True, text=True)

        # Parse Accuracy and AUC from the inference output
        # Search from the end of the output to find the TEST results, not the metadata load
        acc_match = re.search(r"Accuracy\s*:\s*([0-9.]+)", inf_result.stdout.split("Inference Results")[-1])
        auc_match = re.search(r"AUC\s*:\s*([0-9.]+)", inf_result.stdout.split("Inference Results")[-1])
        
        if not acc_match or not auc_match:
            print(f"❌ Failed to parse accuracy or AUC for {ckpt_name}. Skipping.")
            continue

        accuracy = float(acc_match.group(1))
        auc = float(auc_match.group(1))

        print(f"   -> Result: Accuracy {accuracy:.4f} | AUC {auc:.4f}\n")

        results.append({
            "ckpt": ckpt_name,
            "auc": auc,
            "acc": accuracy,
            "temp": temperature,
            "thr": threshold
        })

    # --- Print Final Leaderboard ---
    print("\n\n" + "🏆 MODEL LEADERBOARD 🏆".center(80))
    print("=" * 80)
    print(f"{'Checkpoint Name':<35} | {'AUC':<8} | {'Accuracy':<8} | {'Temp':<6} | {'Threshold':<9}")
    print("-" * 80)
    
    # Sort the results by AUC (highest to lowest)
    results.sort(key=lambda x: x['auc'], reverse=True)

    for r in results:
        print(f"{r['ckpt']:<35} | {r['auc']:.4f}   | {r['acc']:.4f}   | {r['temp']:.4f} | {r['thr']:.4f}")
    print("=" * 80)

    if results:
        best = results[0]
        print(f"\n👑 THE OVERALL CHAMPION IS: {best['ckpt']}")
        print(f"You can now safely delete the others and use this model for your Web UI!")

if __name__ == '__main__':
    main()