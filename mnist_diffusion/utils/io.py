import os
import torch
from torchvision import utils


def save_grid(samples, path: str, nrow: int = 4):
    """
    samples: (B,1,28,28) in [-1,1]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = utils.make_grid((samples.clamp(-1, 1) + 1) / 2, nrow=nrow)
    utils.save_image(grid, path)


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
