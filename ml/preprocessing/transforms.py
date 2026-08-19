import cv2
import torch

from ml.config import Config


class FrameTransform:

    def __init__(self):

        self.image_size = Config.IMAGE_SIZE

    def __call__(
        self,
        frame
    ):

        if frame is None:

            raise ValueError(
                "Input frame is None."
            )

        # =====================================================
        # OpenCV:
        #
        # BGR
        #
        # Training pipeline:
        #
        # BGR -> RGB
        # =====================================================

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =====================================================
        # Resize
        #
        # Training:
        #
        # 288 x 288
        # =====================================================

        frame = cv2.resize(
            frame,
            (
                self.image_size,
                self.image_size
            )
        )

        # =====================================================
        # numpy uint8
        #
        # [H, W, C]
        #
        # ->
        #
        # float32
        # =====================================================

        frame = torch.tensor(
            frame,
            dtype=torch.float32
        )

        # =====================================================
        # HWC -> CHW
        #
        # [288, 288, 3]
        #
        # ->
        #
        # [3, 288, 288]
        # =====================================================

        frame = frame.permute(
            2,
            0,
            1
        )

        # =====================================================
        # [0, 255] -> [0, 1]
        #
        # EXACTLY like training
        # =====================================================

        frame /= 255.0

        return frame


def build_frame_transform():

    return FrameTransform()