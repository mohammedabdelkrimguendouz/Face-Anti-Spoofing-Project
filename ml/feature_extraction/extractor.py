import torch

from ml.config import Config

from ml.preprocessing.transforms import (
    build_frame_transform
)

from ml.feature_extraction.efficientnet import (
    load_efficientnet
)


class FeatureExtractor:

    def __init__(
        self,
        model_path=None,
        device=None,
        batch_size=None
    ):

        # =====================================================
        # Device
        # =====================================================

        if device is None:

            device = Config.DEVICE

        self.device = torch.device(
            device
        )

        # =====================================================
        # Model path
        # =====================================================

        if model_path is None:

            model_path = (
                Config.EFFICIENTNET_PATH
            )

        # =====================================================
        # Batch size
        # =====================================================

        if batch_size is None:

            batch_size = (
                Config.FEATURE_BATCH_SIZE
            )

        self.batch_size = batch_size

        # =====================================================
        # Preprocessing
        # =====================================================

        self.transform = (
            build_frame_transform()
        )

        # =====================================================
        # Load YOUR EfficientNet-B2
        # =====================================================

        self.model = load_efficientnet(
            model_path=model_path,
            device=self.device
        )

        print(
            "EfficientNet-B2 loaded successfully."
        )

        print(
            f"Device       : {self.device}"
        )

        print(
            f"Feature size : {Config.FEATURE_SIZE}"
        )

    # =========================================================
    # Single frame
    # =========================================================

    @torch.no_grad()
    def extract_frame(
        self,
        frame
    ):

        # -----------------------------------------------------
        # Preprocess
        # -----------------------------------------------------

        tensor = self.transform(
            frame
        )

        # -----------------------------------------------------
        # Add batch dimension
        #
        # [3, 288, 288]
        #
        # ->
        #
        # [1, 3, 288, 288]
        # -----------------------------------------------------

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # -----------------------------------------------------
        # Extract
        # -----------------------------------------------------

        features = self.model(
            tensor
        )

        # -----------------------------------------------------
        # [1, 1408]
        #
        # ->
        #
        # [1408]
        # -----------------------------------------------------

        return features.squeeze(
            0
        ).cpu()

    # =========================================================
    # Multiple frames
    # =========================================================

    @torch.no_grad()
    def extract(
        self,
        frames
    ):

        if len(frames) == 0:

            raise ValueError(
                "No frames were provided."
            )

        tensors = []

        # =====================================================
        # Preprocess frames
        # =====================================================

        for frame in frames:

            tensor = self.transform(
                frame
            )

            tensors.append(
                tensor
            )

        # =====================================================
        # Batch inference
        # =====================================================

        features = []

        for start in range(
            0,
            len(tensors),
            self.batch_size
        ):

            batch_tensors = tensors[
                start:
                start + self.batch_size
            ]

            batch = torch.stack(
                batch_tensors
            )

            batch = batch.to(
                self.device
            )

            batch_features = self.model(
                batch
            )

            features.append(
                batch_features.cpu()
            )

        # =====================================================
        # Concatenate
        #
        # [B1, 1408]
        # [B2, 1408]
        # ...
        #
        # ->
        #
        # [N, 1408]
        # =====================================================

        features = torch.cat(
            features,
            dim=0
        )

        return features