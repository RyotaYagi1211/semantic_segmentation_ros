import numpy as np
import torch
import torch.nn.functional as F

from semantic_segmentation_ros.networks import load_model


class SemanticSegmentation:
    def __init__(self, model_name: str, encoder_name: str, encoder_weights: str,
                 in_channels: int, classes: int, model_path: str, device: str):
        self.device = device
        self.model = load_model(model_name, encoder_name, encoder_weights,
                                in_channels, classes, model_path, self.device)
        self.model.eval()

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Predict the segmentation mask for a given RGB image.
        Args:
            img (np.ndarray): numpy array of RGB image with shape (C, H, W).
        Returns:
            np.ndarray: predicted segmentation mask (H, W).
        """
        
#padあり
        # H, W = img.shape[1], img.shape[2]
        # # --- pad (学習と同じく下に8pxだけ) ---
        # img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        # img_t = F.pad(img_t, (0, 0, 0, 8))  # (left, right, top, bottom)

        # # --- predict ---
        # with torch.no_grad():
        #     y_pred = self.model(img_t)
        #     y_pred = y_pred.argmax(1)

        # # --- unpad (元の1080に戻す) ---
        # y_pred = y_pred[:, :H, :]
        # return y_pred.cpu().squeeze(0).numpy()

# #padなし
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)

                # 推論
        with torch.no_grad():
            y_pred = self.model(img_t)
            y_pred = y_pred.argmax(1)

                # torch → numpy
        return y_pred.cpu().squeeze(0).numpy()