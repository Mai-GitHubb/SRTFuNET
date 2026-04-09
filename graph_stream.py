import torch
import torch.nn as nn


class LandmarkExtractor(nn.Module):
    """
    MLP-based facial landmark feature extractor.

    Input : (B, 1405) — 468 MediaPipe landmarks × (x, y, z) + 1 detection flag
    Output: (B, 512)

    Change vs previous version:
        - input_dim raised from 1404 → 1405 to accommodate the binary
          "face detected" flag appended by dataset.py.
          The flag (1.0 = detected, 0.0 = not detected) lets this module
          learn to down-weight no-detection samples instead of treating the
          neutral-filled coordinate vector as a meaningful face.

    Architecture (unchanged):
        - Main branch: 1405 → 1024 → 512 with BN, ReLU, Dropout
        - Skip branch: 1405 → 512 (linear projection)
        - LayerNorm after residual add
    """

    def __init__(self, input_dim=1405, hidden_dim=1024, output_dim=512, dropout=0.2):
        super(LandmarkExtractor, self).__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim

        # Main branch: 1405 → 1024 → 512
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

        # Residual / skip branch: 1405 → 512 (linear projection only)
        self.skip = nn.Linear(input_dim, output_dim, bias=False)

        # Final layer norm after the residual add
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1405) raw landmark coordinates + detection flag
        Returns:
            features: (B, 512)
        """
        out = self.main(x) + self.skip(x)   # residual add
        out = self.norm(out)
        return out


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    extractor = LandmarkExtractor()
    extractor.eval()

    total = sum(p.numel() for p in extractor.parameters())
    print(f"LandmarkExtractor | total parameters: {total:,}")

    dummy = torch.randn(2, 1405)
    with torch.no_grad():
        out = extractor(dummy)
    print(f"Output shape: {out.shape}")   # Expected: torch.Size([2, 512])
