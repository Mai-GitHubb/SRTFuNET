import torch
import torch.nn as nn
import timm


class SpatialExtractor(nn.Module):
    """
    Xception-based spatial feature extractor.
    Outputs: (B, 2048)

    Freeze strategy V2 — restored (same as the run that achieved AUC=0.838):
        Unfreeze blocks.10, blocks.11, conv3, conv4, bn3, bn4.
        Everything else frozen.

        Previous experiments showed:
        - Fully frozen  → AUC 0.68  (insufficient feature adaptation)
        - blocks.11 only → AUC 0.68  (too little fine-tuning)
        - blocks.10+11   → AUC 0.838 (best result — restored here)
        - All unfrozen   → AUC 0.826 then regression (overfitting)
    """

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=True, num_classes=0)

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze blocks 8 and 9 to allow for mid-level feature adaptation
        for name, param in self.model.named_parameters():
            if any(k in name for k in ('blocks.8', 'blocks.9', 'blocks.10', 'blocks.11',
                                        'conv3', 'conv4', 'bn3', 'bn4')):
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    extractor = SpatialExtractor()
    extractor.eval()
    trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in extractor.parameters() if not p.requires_grad)
    print(f"SpatialExtractor V2 | trainable: {trainable:,} | frozen: {frozen:,}")
    with torch.no_grad():
        out = extractor(torch.randn(2, 3, 299, 299))
    print(f"Output shape: {out.shape}")