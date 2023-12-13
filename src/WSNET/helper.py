"""
In this file, methods that are used in multiple of the scripts are collected to make the necessary adaptions only once
and to be able to import methods and make the code more readable.
Commented code from the source is removed.
"""
import os

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from src.helper import get_data_dirs


def assign_patches(x):
    image, patch = x
    image = patch
    return image


class CreatePatches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(CreatePatches, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        patches = []
        # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
        input_image_size = inputs.shape[1]
        for i in range(0, input_image_size, self.patch_size):
            for j in range(0, input_image_size, self.patch_size):
                patches.append(inputs[:, i : i + self.patch_size, j : j + self.patch_size, :])
        return patches


def putconcate(x, layer_count=4):
    if layer_count == 4:
        x1, x2, x3, x4 = x
        return K.concatenate([x1, x2, x3, x4], axis=2)
    elif layer_count == 3:
        x1, x2, x3 = x
        return K.concatenate([x1, x2, x3], axis=2)


def putconcate_vert(x, layer_count=4):
    if layer_count == 4:
        x1, x2, x3, x4 = x
        return K.concatenate([x1, x2, x3, x4], axis=1)
    elif layer_count == 3:
        x1, x2, x3 = x
        return K.concatenate([x1, x2, x3], axis=1)


def putall(x, layer_count=9):
    if layer_count == 9:
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        return K.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9], axis=1)
    elif layer_count == 16:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = x
        return K.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], axis=1)


def merge_patches(x):
    return K.reshape(x, (-1, 192, 192, 1))


def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        A.RandomRotate90(),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.Blur(blur_limit=3),
        A.RandomBrightnessContrast(p=0.2),
        # A.RGBShift(p=0.2),
        # A.HueSaturationValue(),
        A.Transpose(),
    ],
    additional_targets={"image0": "image"},
)


def generate_data(images_list, batch_size, dims, train=False, val=False, test=False, two_inputs=False):
    """Replaces Keras' native ImageDataGenerator."""
    data_dir, mask_dir = get_data_dirs(colab=False)
    if not (train or val or test):
        raise ValueError("one of train or val or test need to be True")
    i = 0
    while True:
        image_batch = []
        image_batch1 = []
        label_batch = []
        for b in range(batch_size):
            if i == len(images_list):
                i = 0
            if i < len(images_list):
                # get the filenames
                sample_image_filename = images_list[i].decode("utf-8")
                sample_label_filename = images_list[i].decode("utf-8")

                # read the image
                image = cv2.imread(os.path.join(data_dir, sample_image_filename), 1)
                image = cv2.resize(image, dims)

                # read the lable
                label = cv2.imread(os.path.join(mask_dir, sample_label_filename), 0)
                label = cv2.resize(label, dims)
                label = np.expand_dims(label, axis=2)

                if not test:  # do not transform test images
                    # transform images
                    transformed = transform(image=image, image0=label)
                    aug_img = transformed["image"]  # transformed wound images
                    aug_mask = transformed["image0"]  # transformed mask

                    # append augmented images to lists
                    image_batch.append(aug_img)
                    image_batch1.append(aug_img)
                    label_batch.append(aug_mask)
                else:  # for test images
                    # append original images to lists
                    image_batch.append(image)
                    image_batch1.append(image)
                    label_batch.append(label)
            i += 1

        # normalize per batch
        if image_batch and label_batch:
            image_batch = normalize(np.array(image_batch))
            image_batch1 = normalize(np.array(image_batch1))
            label_batch = normalize(np.array(label_batch))
            if not two_inputs:
                yield image_batch, label_batch
            else:
                yield (image_batch, image_batch1), label_batch
