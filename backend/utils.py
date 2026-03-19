import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert RGB image into 9-channel tensor: RGB + HSV + LAB."""
    img = image.resize((512, 512))
    img_np = np.array(img)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)

    combined = np.concatenate([img_np, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1) / 255.0

    return torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
