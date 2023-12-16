import os
from math import ceil
from pathlib import Path

from matplotlib import pyplot as plt


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


def plot_first_batch(model, test_gen, two_inputs=False, batch_size=16):
    for image, label in test_gen:
        res = model.predict(image)
        num_rows = ceil(batch_size / 2)
        plt.subplots(num_rows, 6, figsize=(12, 16))
        for i in range(batch_size):
            plt.subplot(num_rows, 6, i * 3 + 1)
            if two_inputs:
                plt.imshow(image[0][i, :, :, :])
            else:
                plt.imshow(image[i, :, :, :])
            plt.axis("off")
            plt.subplot(num_rows, 6, i * 3 + 2)
            plt.imshow(res[i, :, :, :], cmap="gray")
            plt.axis("off")
            plt.subplot(num_rows, 6, i * 3 + 3)
            plt.imshow(label[i, :, :, :], cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        break
