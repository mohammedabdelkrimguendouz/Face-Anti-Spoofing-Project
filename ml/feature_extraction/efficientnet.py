import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2


class EfficientNetB2FeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        # ==========================================
        # EfficientNet-B2 architecture
        # ==========================================

        model = efficientnet_b2(
            weights=None
        )

        self.features = model.features
        self.avgpool = model.avgpool

        # ==========================================
        # ImageNet normalization
        # ==========================================

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "mean",
            mean
        )

        self.register_buffer(
            "std",
            std
        )

    def forward(self, x):

        # ==========================================
        # Normalize
        # ==========================================

        x = (
            x - self.mean
        ) / self.std

        # ==========================================
        # EfficientNet features
        # ==========================================

        x = self.features(x)

        # ==========================================
        # Global Average Pooling
        # ==========================================

        x = self.avgpool(x)

        # ==========================================
        # Flatten
        # ==========================================

        x = torch.flatten(
            x,
            1
        )

        # Output:
        # (B, 1408)

        return x


def load_efficientnet(
    model_path,
    device
):

    device = torch.device(
        device
    )

    # ==========================================
    # Load checkpoint
    # ==========================================

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # ==========================================
    # Create architecture
    # ==========================================

    model = EfficientNetB2FeatureExtractor()

    # ==========================================
    # Handle .pt checkpoint
    # ==========================================

    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    # ==========================================
    # Handle .pth state_dict
    # ==========================================

    else:

        state_dict = checkpoint

    # ==========================================
    # Load weights
    # ==========================================

    model.load_state_dict(
        state_dict
    )

    # ==========================================
    # Move to device
    # ==========================================

    model = model.to(
        device
    )

    # ==========================================
    # Evaluation mode
    # ==========================================

    model.eval()

    # ==========================================
    # Freeze
    # ==========================================

    for parameter in model.parameters():

        parameter.requires_grad = False

    return model