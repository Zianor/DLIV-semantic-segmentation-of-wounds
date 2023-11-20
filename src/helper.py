import os
from pathlib import Path


def get_data_dirs(colab=False):
    """Return the data and masks directory"""
    if colab:
        data_dir = "/content/drive/MyDrive/DLIV/data/wound_images"
        mask_dir = "/content/drive/MyDrive/DLIV/data/wound_masks"
    else:
        data_dir = os.path.join(Path(__file__).parent, "data/wound_images")
        mask_dir = os.path.join(Path(__file__).parent, "data/wound_masks")
    return data_dir, mask_dir


def get_checkpoint_path(model_name, colab=False):
    if colab:
        return f"/content/drive/MyDrive/DLIV/wacv/{model_name}.h5"
    else:
        return os.path.join(Path(__file__).parent, "wacv", f"{model_name}.h5")
