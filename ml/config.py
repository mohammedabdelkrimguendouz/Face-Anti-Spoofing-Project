from pathlib import Path
import torch


class Config:

    # =========================================================
    # Project
    # =========================================================

    BASE_DIR = Path(__file__).resolve().parent.parent

    # =========================================================
    # Device
    # =========================================================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================================================
    # EfficientNet-B2
    # =========================================================

    IMAGE_SIZE = 288

    CHANNELS = 3

    FEATURE_SIZE = 1408

    NUM_FRAMES = 150

    FEATURE_BATCH_SIZE = 16

    # =========================================================
    # Saved EfficientNet model
    # =========================================================

    MODEL_DIR = BASE_DIR / "weights"

    EFFICIENTNET_PATH = (
        MODEL_DIR /
        "EfficientNet_B2.pt"
    )

    # =========================================================
    # ImageNet normalization
    #
    # IMPORTANT:
    # These values are used INSIDE the saved
    # EfficientNetFeatureExtractor.
    #
    # Therefore transforms.py must NOT normalize.
    # =========================================================

    NORMALIZE_MEAN = (
        0.485,
        0.456,
        0.406
    )

    NORMALIZE_STD = (
        0.229,
        0.224,
        0.225
    )