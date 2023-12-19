import os
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
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


def get_data_dirs_dfuc(colab=False):
    if colab:
        data_dir = "/content/drive/MyDrive/DLIV/data/DiabetesFootUlcerChallenge2021/wound_images"
        mask_dir = "/content/drive/MyDrive/DLIV/data/DiabetesFootUlcerChallenge2021/wound_masks"
    else:
        data_dir = os.path.join(Path(__file__).parent, "data/DiabetesFootUlcerChallenge2021/wound_images")
        mask_dir = os.path.join(Path(__file__).parent, "data/DiabetesFootUlcerChallenge2021/wound_masks")
    return data_dir, mask_dir


def get_checkpoint_path(model_name, colab=False):
    if colab:
        return f"/content/drive/MyDrive/DLIV/trained_models/{model_name}.h5"
    else:
        return os.path.join(Path(__file__).parent, "trained_models", f"{model_name}.h5")


def plot_first_batch(model, test_gen, two_inputs=False, batch_size=16):
    for image, label in test_gen:
        res = model.predict(image)
        num_rows = ceil(batch_size / 2)
        plt.subplots(num_rows, 6, figsize=(12, 16))
        for i in range(batch_size):
            if i == 0 or i == 1:
                add_title = True
            else:
                add_title = False
            plt.subplot(num_rows, 6, i * 3 + 1)
            if two_inputs:
                plt.imshow(image[0][i, :, :, :])
            else:
                plt.imshow(image[i, :, :, :])
            if add_title:
                plt.title("Wound Image")
            plt.axis("off")
            plt.subplot(num_rows, 6, i * 3 + 2)
            plt.imshow(res[i, :, :, :], cmap="gray")
            plt.axis("off")
            if add_title:
                plt.title("Predicted mask")
            plt.subplot(num_rows, 6, i * 3 + 3)
            plt.imshow(label[i, :, :, :], cmap="gray")
            if add_title:
                plt.title("Ground truth mask")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        break


def get_architecture_title(model_architecture):
    architectures = {"local": "Local model", "global": "Global model", "global-local": "Global-Local model"}
    return architectures[model_architecture]


def get_segmentation_model_title(model):
    models = {"unet": "U-Net", "linknet": "Linknet", "pspnet": "PSPNet", "fpn": "FPN"}
    return models[model]


def create_markdown_table_str_for_metrics(results_dict):
    markdown_lines = []
    table_keys = "| " + " | ".join(f"{key:<5}" for key in results_dict.keys()) + " |"
    markdown_lines.append(table_keys)
    markdown_lines.append("|" + "|".join("-" * max(len(key) + 2, 7) for key in results_dict.keys()) + "|")
    table_values = (
        "| "
        + " | ".join(
            f"{value:.3f}{' ' * (len(key) - 5) if len(key) > 5 else ''}"
            for key, value in results_dict.items()
        )
        + " |"
    )
    markdown_lines.append(table_values)
    markdown_lines.append("")
    return "\n".join(markdown_lines)


def get_dfuc_data(batch_size=16, dims=(192, 192), two_inputs=False):
    from src.WSNET.helper import normalize

    data_dir, mask_dir = get_data_dirs_dfuc(colab=False)

    filenames = [image for image in os.listdir(data_dir) if image != ".gitkeep"]
    i = 0
    while True:
        curr_image_batch = []
        curr_label_batch = []
        for b in range(batch_size):
            if i == len(filenames):
                i = 0
            if i < len(filenames):
                filename = filenames[i]

                # read image
                image = cv2.imread(os.path.join(data_dir, filename), 1)
                image = cv2.resize(image, dims)
                curr_image_batch.append(image)
                # read label
                label = cv2.imread(os.path.join(mask_dir, filename), 0)
                label = cv2.resize(label, dims)
                label = np.expand_dims(label, axis=2)
                curr_label_batch.append(label)
            i += 1
        if curr_image_batch and curr_label_batch:
            # normalize batch
            curr_image_batch = normalize(np.array(curr_image_batch))
            curr_label_batch = normalize(np.array(curr_label_batch))
            if not two_inputs:
                yield curr_image_batch, curr_label_batch
            else:
                yield (curr_image_batch, curr_image_batch), curr_label_batch


def get_dfuc_dataset(two_inputs, batch_size=16, input_size=192):
    if two_inputs:
        output_signature = (
            (
                tf.TensorSpec(shape=(batch_size, input_size, input_size, 3)),
                tf.TensorSpec(shape=(batch_size, input_size, input_size, 3)),
            ),
            tf.TensorSpec(shape=(batch_size, input_size, input_size, 1)),
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(batch_size, 192, 192, 3)),
            tf.TensorSpec(shape=(batch_size, 192, 192, 1)),
        )
    dataset = tf.data.Dataset.from_generator(
        get_dfuc_data,
        args=[batch_size, (input_size, input_size), two_inputs],
        output_signature=output_signature,
    )
    return dataset
