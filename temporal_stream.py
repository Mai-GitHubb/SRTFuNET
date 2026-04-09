import torch
import torch.nn as nn
import torchvision.models.video as video


class TemporalExtractor(nn.Module):
    """
    R(2+1)D-18-based temporal feature extractor.
    Outputs: (B, 512)

    Freeze strategy V2 — restored (same as the run that achieved AUC=0.838):
        Unfreeze layer3 and layer4. Everything else frozen.

        Previous experiments:
        - Fully frozen  → AUC 0.68
        - layer4 only   → AUC 0.68
        - layer3+layer4 → AUC 0.838 (best — restored here)
    """

    def __init__(self):
        super().__init__()
        self.model    = video.r2plus1d_18(weights='DEFAULT')
        self.model.fc = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    extractor = TemporalExtractor()
    extractor.eval()
    trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in extractor.parameters() if not p.requires_grad)
    print(f"TemporalExtractor V2 | trainable: {trainable:,} | frozen: {frozen:,}")
    with torch.no_grad():
        out = extractor(torch.randn(2, 3, 16, 112, 112))
    print(f"Output shape: {out.shape}")