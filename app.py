import os
import cv2
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

# ── Grad-CAM Imports ──────────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Configuration (Calibrated Ensemble Parameters) ────────────────────────────
CHECKPOINT_PATHS = [
    "checkpoints/topk_e07_auc0.9736.pth",
    "checkpoints/topk_e10_auc0.9713.pth",
    "checkpoints/topk_e13_auc0.9680.pth"
]
TEMPERATURE = 1.3630  # Calibrated for ensemble logit smoothing
THRESHOLD = 0.4600    # Optimal balanced threshold for 91.7% Accuracy

# ── Initialization ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ensemble():
    """Loads all three models for collective voting."""
    models = []
    print(f"🚀 Initializing ensemble on {device}...")
    for path in CHECKPOINT_PATHS:
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} missing!")
            continue
            
        model = LateFusionDeepfakeDetector()
        try:
            import numpy._core.multiarray as _npcma
            torch.serialization.add_safe_globals([_npcma.scalar])
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
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
    cap = cv2.VideoCapture(video_path)
    frames, total_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(int(total_frames * 0.1), int(total_frames * 0.9), 16, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()

    if not frames: frames = [np.zeros((112, 112, 3), dtype=np.uint8)] * 16
    while len(frames) < 16: frames.append(frames[-1])
    mid_rgb = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_BGR2RGB)
    
    spatial_frame = cv2.resize(mid_rgb, (299, 299))
    spatial_tensor = TF.normalize(torch.from_numpy(spatial_frame).float().permute(2,0,1)/255.0, [0.5,0.5,0.5], [0.5,0.5,0.5]).unsqueeze(0)

    t_frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (112, 112)) for f in frames]
    temporal_tensor = torch.from_numpy(np.stack(t_frames)).float().permute(3,0,1,2) / 255.0
    mean, std = torch.tensor([0.432, 0.394, 0.376]).view(3,1,1,1), torch.tensor([0.228, 0.221, 0.216]).view(3,1,1,1)
    temporal_tensor = ((temporal_tensor - mean) / std).unsqueeze(0)

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        res = face_mesh.process(mid_rgb)
        if res.multi_face_landmarks:
            lms = [val for lm in res.multi_face_landmarks[0].landmark for val in (lm.x, lm.y, lm.z)]
            lm_arr = np.append(np.array(lms, dtype=np.float32), 1.0)
        else:
            lm_arr = np.append(np.full(468*3, 0.5, dtype=np.float32), 0.0)
    
    return spatial_tensor, temporal_tensor, torch.from_numpy(lm_arr).float().unsqueeze(0), spatial_frame

# ── Analysis Logic ────────────────────────────────────────────────────────────
def predict_video(video_filepath):
    try:
        if video_filepath is None:
            raise ValueError("❌ No video uploaded. Please upload a video file to analyze.")
        
        # Extract filename from file path
        import os
        filename = os.path.basename(video_filepath)
        
        spatial, temporal, landmark, base_img_rgb = extract_and_preprocess(video_filepath)
        spatial, temporal, landmark = spatial.to(device), temporal.to(device), landmark.to(device)
        
        # Ensemble Prediction
        total_fake_prob = 0
        with torch.no_grad():
            for m in ensemble_models:
                logits = m(spatial, temporal, landmark)
                total_fake_prob += torch.softmax(logits / TEMPERATURE, dim=1)[0][1].item()
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
            analyze_btn.click(fn=predict_video, inputs=video_input, outputs=[result_md, prob_slider, heatmap_out, landmark_out, video_display])

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