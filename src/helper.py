import os
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.WSNET.helper import generate_data


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


def split_train_test_validation():
    """
    :return: [train_images, validation_images, test_images]
    """
    data_dir, mask_dir = get_data_dirs(False)
    all_images = os.listdir(data_dir)

    train_images, test_images = train_test_split(all_images, train_size=0.7, test_size=0.3, random_state=0)
    test_images, validation_images = train_test_split(
        test_images, train_size=0.5, test_size=0.5, random_state=0
    )
    return train_images, validation_images, test_images


def get_datasets(train_images, validation_images, test_images, two_inputs, batch_size=16, input_size=192):
    """Transform lists of filenames into tensorflow datasets
    :param train_images: list of filenames that are part of the training set
    :param validation_images: list of filenames that are part of the validation set
    :param test_images: list of filenames that are part of the test set
    :param two_inputs: if True, input of model is two images, else 1 image
    :param batch_size: batch size
    :param input_size: input size of the images, must be square images
    :returns: train dataset, validation dataset, test_dataset
    :rtype: [tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
    """
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

    train_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[train_images, batch_size, (input_size, input_size), True, False, False, True],
        output_signature=output_signature,
    )
    val_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[validation_images, batch_size, (input_size, input_size), False, True, False, True],
        output_signature=output_signature,
    )
    test_gen = tf.data.Dataset.from_generator(
        generate_data,
        args=[test_images, batch_size, (input_size, input_size), False, False, True, True],
        output_signature=output_signature,
    )
    return train_gen, val_gen, test_gen
