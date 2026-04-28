import os
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms.functional as TF
import mediapipe as mp
import gradio as gr
import warnings
from PIL import Image

# ── Warning Suppression ───────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from late_fusion_model import LateFusionDeepfakeDetector
from dataset import LateFusionDataset  # ⬅️ IMPORT THE DATASET TO ENSURE 1:1 PARITY

# ── Grad-CAM Imports ──────────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Configuration (Calibrated Ensemble Parameters) ────────────────────────────
TEMPERATURE = 1.3630  
THRESHOLD = 0.4600    

# ── Initialization ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ensemble():
    """Loads all models using the same glob logic as inference.py."""
    models = []
    print(f"🚀 Initializing ensemble on {device}...")
    
    # ⬅️ Matches how inference.py dynamically finds ensemble checkpoints
    topk_paths = sorted(glob.glob('checkpoints/topk_*.pth'))
    if not topk_paths:
        print("No top-k checkpoints found. Falling back to best_auc.pth and best_balanced.pth.")
        topk_paths = [p for p in ['checkpoints/best_auc.pth', 'checkpoints/best_balanced.pth'] if os.path.exists(p)]
        
    for path in topk_paths:
        if not os.path.exists(path):
            continue
            
        model = LateFusionDeepfakeDetector()
        try:
            import numpy._core.multiarray as _npcma
            torch.serialization.add_safe_globals([_npcma.scalar])
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=device, weights_only=False)

        # ⬅️ Match state_dict extraction logic exactly
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
            
        model.to(device).eval()
        models.append(model)
        print(f" ✅ Loaded: {os.path.basename(path)}")
    return models

ensemble_models = load_ensemble()

# ── Grad-CAM / XAI Wrapper ────────────────────────────────────────────────────
class SpatialWrapper(torch.nn.Module):
    def __init__(self, model, temporal, landmark):
        super().__init__()
        self.model = model
        self.temporal = temporal
        self.landmark = landmark
        
    def forward(self, spatial):
        logits = self.model(spatial, self.temporal, self.landmark)
        return logits / max(TEMPERATURE, 1e-3)

def visualize_landmark_artifacts(base_img, landmark_tensor, model, target_class):
    """Identifies and marks the top 15 most suspicious facial landmarks in red."""
    landmark_tensor = landmark_tensor.clone().detach().to(device)
    landmark_tensor.requires_grad = True
    
    # Forward pass through landmark stream and classifier
    feat = model.landmark_extractor(landmark_tensor)
    # Concatenate features to simulate the 1536-d input the classifier expects
    logits = model.classifier(torch.cat([feat, feat, feat], dim=1))
    
    model.zero_grad()
    loss = logits[0][target_class]
    loss.backward()
    
    grads = landmark_tensor.grad.abs().squeeze().cpu().numpy()[:-1] # Exclude flag
    point_grads = grads.reshape(468, 3).sum(axis=1)
    top_indices = np.argsort(point_grads)[-15:]
    
    marked_img = base_img.copy()
    h, w, _ = marked_img.shape
    coords = landmark_tensor.detach().cpu().numpy().squeeze()[:-1].reshape(468, 3)
    
    for idx in top_indices:
        cx, cy = int(coords[idx][0] * w), int(coords[idx][1] * h)
        cv2.circle(marked_img, (cx, cy), 4, (255, 0, 0), -1) 
        cv2.circle(marked_img, (cx, cy), 6, (255, 0, 0), 1)
    return marked_img

# ── Preprocessing ─────────────────────────────────────────────────────────────
def extract_and_preprocess(video_path):
    """
    Delegates tensor extraction to LateFusionDataset to guarantee identical 
    preprocessing logic to inference scripts.
    """
    # 1. Exact dataset extraction for prediction
    dataset = LateFusionDataset(
        video_paths=[video_path],
        labels=[0], # Dummy label
        is_training=False,
        num_frames=16
    )
    
    batch = dataset[0]
    spatial_tensor = batch['spatial'].unsqueeze(0)
    temporal_tensor = batch['temporal'].unsqueeze(0)
    landmark_tensor = batch['landmark'].unsqueeze(0)

    # 2. Separate middle frame extraction purely for visual XAI heatmap background
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            mid_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            mid_rgb = np.zeros((299, 299, 3), dtype=np.uint8)
    else:
        mid_rgb = np.zeros((299, 299, 3), dtype=np.uint8)
    cap.release()
    
    spatial_frame = cv2.resize(mid_rgb, (299, 299))
    
    return spatial_tensor, temporal_tensor, landmark_tensor, spatial_frame

# ── Analysis Logic ────────────────────────────────────────────────────────────
def predict_video(video_filepath, n_tta):
    try:
        if video_filepath is None:
            raise ValueError("❌ No video uploaded. Please upload a video file to analyze.")

        filename = os.path.basename(video_filepath)
        
        spatial, temporal, landmark, base_img_rgb = extract_and_preprocess(video_filepath)
        spatial, temporal, landmark = spatial.to(device), temporal.to(device), landmark.to(device)
        
        # Ensemble Prediction
        total_fake_prob = 0
        with torch.no_grad():
            for m in ensemble_models:
                model_prob = 0
                for i in range(int(n_tta)):
                    s_input = spatial
                    if i == 1: # Horizontal flip for second TTA pass
                        s_input = torch.flip(spatial, dims=[-1])
                    
                    logits = m(s_input, temporal, landmark)
                    probs = torch.softmax(logits / TEMPERATURE, dim=1)[0][1].item()
                    model_prob += probs
                total_fake_prob += (model_prob / int(n_tta))
        fake_prob = total_fake_prob / len(ensemble_models)
        
        res_label = "🚨 FAKE" if fake_prob >= THRESHOLD else "✅ REAL"
        color = "#ff4b4b" if fake_prob >= THRESHOLD else "#00d1b2"
        classification_text = f"<h2 style='color: {color}; text-align: center;'>{res_label} (Confidence: {fake_prob*100:.1f}%)</h2>"
        
        # XAI Visuals
        target_class = 1 if fake_prob >= THRESHOLD else 0
        wrapper = SpatialWrapper(ensemble_models[0], temporal, landmark)
        cam = GradCAM(model=wrapper, target_layers=[ensemble_models[0].spatial_extractor.model.conv4])
        grayscale_cam = cam(input_tensor=spatial, targets=[ClassifierOutputTarget(target_class)])[0, :]
        heatmap_viz = show_cam_on_image(base_img_rgb.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
        landmark_marks = visualize_landmark_artifacts(base_img_rgb, landmark, ensemble_models[0], target_class)
            
        return classification_text, fake_prob, heatmap_viz, landmark_marks, filename
    except Exception as e:
        return f"## Error: {str(e)}", 0, None, None, "Error"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="STG-FuNET Forensic Suite") as demo:
    gr.HTML("<h1 style='text-align: center;'>🕵️‍♂️ STG-FuNET for High-Fidelity Deepfake Forensic Detection</h1>")
    
    with gr.Tabs():
        with gr.TabItem("Forensic Analysis"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(label="Input Evidence (Video)", file_types=["video"])
                    video_display = gr.Textbox(label="Uploaded Video", interactive=False, value="No video uploaded")
                    # ⬅️ TTA default updated to 1 to match your evaluation scripts
                    tta_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="TTA Passes (Test-Time Augmentation)")
                    analyze_btn = gr.Button("🔍 Run Multi-Stream Analysis", variant="primary")
                    gr.Markdown("### Ensemble Core Configuration")
                    with gr.Row():
                        gr.Number(value=TEMPERATURE, label="Calibrated Temp", interactive=False)
                        gr.Number(value=THRESHOLD, label="Decision Threshold", interactive=False)
                with gr.Column():
                    result_md = gr.HTML("<h3 style='text-align: center;'>Awaiting forensic input...</h3>")
                    prob_slider = gr.Slider(0, 1, label="Probability Map (Fake Density)", interactive=False)
                    with gr.Row():
                        heatmap_out = gr.Image(label="Spatial Artifacts (Heatmap)")
                        landmark_out = gr.Image(label="Geometric Artifacts (Red Marks)")
            analyze_btn.click(fn=predict_video, inputs=[video_input, tta_slider], outputs=[result_md, prob_slider, heatmap_out, landmark_out, video_display])

        with gr.TabItem("Performance Benchmarks"):
            gr.Markdown("## 📊 Ensemble Integrity Report")
            with gr.Row():
                metrics_img = "results/inference_metrics.png"
                if os.path.exists(metrics_img):
                    gr.Image(value=metrics_img, label="Confusion Matrix & ROC Analytics")
                else:
                    gr.Markdown("> ⚠️ **Benchmark Data Missing:** Run `inference.py` to generate `inference_metrics.png`.")
            
            gr.Markdown("### Core Performance Metrics")
            with gr.Row():
                gr.Number(value=0.9705, label="Test Set AUC", interactive=False)
                gr.Number(value=0.9170, label="Overall Accuracy", interactive=False)
                gr.Number(value=0.9514, label="Precision Score", interactive=False)
                gr.Number(value=0.9206, label="Recall Score", interactive=False)
                gr.Number(value=0.9357, label="F1-Score", interactive=False)
            
            gr.Markdown("### Per-Class Accuracy")
            with gr.Row():
                gr.Number(value=0.9101, label="Real Video Accuracy", interactive=False)
                gr.Number(value=0.9206, label="Fake Video Accuracy", interactive=False)
            
            gr.Markdown("### Detailed Classification Report")
            gr.HTML("""<div style='width: 100%;overflow-x: auto;'>
            <table style='width: 100%; border-collapse: collapse; margin: 0 auto;'>
            <tr>
            <th style='border: 1px solid white; padding: 12px; text-align: left; color: white;'>Class</th>
            <th style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>Precision</th>
            <th style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>Recall</th>
            <th style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>F1-Score</th>
            <th style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>Support</th>
            </tr>
            <tr>
            <td style='border: 1px solid white; padding: 12px; color: white;'><b>Real</b></td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.86</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.91</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.88</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>178</td>
            </tr>
            <tr>
            <td style='border: 1px solid white; padding: 12px; color: white;'><b>Fake</b></td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.95</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.94</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>340</td>
            </tr>
            <tr>
            <td style='border: 1px solid white; padding: 12px; color: white;'><b>Macro Avg</b></td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.90</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.91</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>518</td>
            </tr>
            <tr>
            <td style='border: 1px solid white; padding: 12px; color: white;'><b>Weighted Avg</b></td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>518</td>
            </tr>
            <tr>
            <td style='border: 1px solid white; padding: 12px; color: white;'><b>Overall Accuracy</b></td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>-</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>-</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>0.92</td>
            <td style='border: 1px solid white; padding: 12px; text-align: center; color: white;'>518</td>
            </tr>
            </table>
            </div>""")
            
if __name__ == "__main__":
    demo.launch(share=False)