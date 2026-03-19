from pathlib import Path
import os
import urllib.request

import torch
import torch.nn as nn
from torchvision import models


def _ensure_weights(weights_file: Path, weights_url: str | None) -> None:
    if weights_file.exists():
        return

    if not weights_url:
        raise FileNotFoundError(f"Model weights not found: {weights_file}")

    weights_file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(weights_url, weights_file)


def load_model(weights_path: str = "weights/best_densenet_9ch.pth") -> torch.nn.Module:
    """Build and load the 9-channel DenseNet model."""
    model = models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    env_path = os.getenv("MODEL_WEIGHTS_PATH", "").strip()
    weights_file = Path(env_path or weights_path)
    weights_url = os.getenv("MODEL_WEIGHTS_URL", "").strip() or None

    _ensure_weights(weights_file, weights_url)

    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
