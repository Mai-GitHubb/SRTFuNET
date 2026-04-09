import torch
import torch.nn as nn
from spatial_stream import SpatialExtractor
from temporal_stream import TemporalExtractor
from graph_stream import LandmarkExtractor


class LateFusionDeepfakeDetector(nn.Module):
    """
    Change vs previous version:
        - LandmarkExtractor now uses input_dim=1405 (was 1404) to accept
          the binary face-detection flag appended by dataset.py.
          All other architecture details are unchanged.
    """

    def __init__(self):
        super(LateFusionDeepfakeDetector, self).__init__()

        # ── Three feature extractors ──────────────────────────────────────────
        self.spatial_extractor  = SpatialExtractor()              # outputs (B, 2048)
        self.temporal_extractor = TemporalExtractor()             # outputs (B, 512)
        self.landmark_extractor = LandmarkExtractor(input_dim=1405)  # outputs (B, 512)

        # ── Project all streams to a common 512-d space ───────────────────────
        self.proj_spatial  = nn.Linear(2048, 512)
        self.proj_temporal = nn.Linear(512,  512)
        # landmark already 512 — no projection needed

        # ── Cross-stream Transformer (3 tokens: spatial, temporal, landmark) ──
        # Increase layers and feedforward dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1536, # Increased from 1024
            dropout=0.15,         # Slightly increased
            batch_first=True,
            norm_first=True
        )
        self.cross_stream_attn = nn.TransformerEncoder(encoder_layer, num_layers=4) # Change 2 -> 4

        # ── Classifier head ───────────────────────────────────────────────────
        # 3 tokens × 512 = 1536 input features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        # ── Weight init for new layers ────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in [self.proj_spatial, self.proj_temporal]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spatial_img, temporal_seq, landmark_arr):
        # ── Extract features ──────────────────────────────────────────────────
        s_feat = self.spatial_extractor(spatial_img)    # (B, 2048)
        t_feat = self.temporal_extractor(temporal_seq)  # (B, 512)
        l_feat = self.landmark_extractor(landmark_arr)  # (B, 512)

        # ── Project to common dim ─────────────────────────────────────────────
        s = self.proj_spatial(s_feat)   # (B, 512)
        t = self.proj_temporal(t_feat)  # (B, 512)
        l = l_feat                       # (B, 512)

        # ── Stack as a 3-token sequence and run cross-stream attention ─────────
        tokens   = torch.stack([s, t, l], dim=1)    # (B, 3, 512)
        attended = self.cross_stream_attn(tokens)   # (B, 3, 512)

        # ── Flatten and classify ──────────────────────────────────────────────
        flat   = attended.reshape(attended.size(0), -1)  # (B, 1536)
        output = self.classifier(flat)                   # (B, 2)

        return output


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = LateFusionDeepfakeDetector()
    model.eval()
    print("LateFusionDeepfakeDetector initialised successfully.")

    spatial_input  = torch.randn(2, 3, 299, 299)
    temporal_input = torch.randn(2, 3, 16, 112, 112)
    landmark_input = torch.randn(2, 1405)           # note: 1405 now

    with torch.no_grad():
        out = model(spatial_input, temporal_input, landmark_input)

    print(f"Output shape: {out.shape}")   # Expected: torch.Size([2, 2])
    print("Smoke-test passed.")
