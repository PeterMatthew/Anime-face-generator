import torch
from torchvision import utils
import math

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optmizer_state_dict"])
    epoch = checkpoint["epoch"]
    stage = checkpoint["stage"]
    alpha = checkpoint["alpha"]

    return epoch, stage, alpha

def save_image(images, n, path, nrow=6):
    utils.save_image(
        images[:n],
        path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1)
    )

def resolution_to_index(resolution):
    return int(math.log2(resolution) - 1)
