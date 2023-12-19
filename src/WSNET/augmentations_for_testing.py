from math import ceil

import tensorflow as tf


def embed_in_black_background(image, mask, proportion_of_original=0.66):
    """Resize the image to 66% (or another value as specified in proportion_of_original of its original size and fill
    the remaining space with black
    :param image: either a single image (as tensor) or a tuple, can be sent in batches
    :param mask: the corresponding mask, can be sent in batches
    :param proportion_of_original: value between 0 and 1, proportion of the original size the resized image should have,
    default is 0.66
    :returns: either transformed (image, image), mask or image, mask
    """
    if isinstance(image, tuple):
        image = image[0]
        two_inputs = True
    else:
        two_inputs = False
    original_size = image.shape[1]
    smaller_size = ceil(original_size * proportion_of_original)
    if smaller_size % 2 == 1:
        smaller_size += 1
    border_width = ceil((original_size - smaller_size) / 2)
    image_smaller = tf.image.resize(image, (smaller_size, smaller_size))
    mask_smaller = tf.image.resize(mask, (smaller_size, smaller_size))
    transformed_image = tf.image.pad_to_bounding_box(
        image_smaller,
        offset_height=border_width,
        offset_width=border_width,
        target_width=original_size,
        target_height=original_size,
    )
    transformed_mask = tf.image.pad_to_bounding_box(
        mask_smaller,
        offset_height=border_width,
        offset_width=border_width,
        target_width=original_size,
        target_height=original_size,
    )
    if two_inputs:
        return (transformed_image, transformed_image), transformed_mask
    return transformed_image, transformed_mask


def adjust_brightness(image, mask, brightness_delta=0.1):
    """Adjust the brightness of the image by `brightness_delta`
    :param image: either a single image (as tensor) or a tuple, can be sent in batches
    :param mask: the corresponding mask, can be sent in batches, is not transformed
    :param brightness_delta: the delta by which the brightness is changed, should be in range (-1, 1), default is 0.1
    :returns: either transformed (image, image), mask or image, mask
    """
    if isinstance(image, tuple):
        image = image[0]
        two_inputs = True
    else:
        two_inputs = False
    image_adjusted_brightness = tf.image.adjust_brightness(image, delta=brightness_delta)
    if two_inputs:
        return (image_adjusted_brightness, image_adjusted_brightness), mask
    else:
        return image_adjusted_brightness, mask


def adjust_contrast(image, mask, contrast_factor=2):
    """Adjust the contrast of the image by `contrast_factor`
    :param image: either a single image (as tensor) or a tuple, can be sent in batches
    :param mask: the corresponding mask, can be sent in batches, is not transformed
    :param contrast_factor: the factor by which the contrast is changed, should be in range (-inf, inf), default is 2
    :returns: either transformed (image, image), mask or image, mask
    """
    if isinstance(image, tuple):
        image = image[0]
        two_inputs = True
    else:
        two_inputs = False
    image_adjusted_contrast = tf.image.adjust_contrast(image, contrast_factor=contrast_factor)
    if two_inputs:
        return (image_adjusted_contrast, image_adjusted_contrast), mask
    else:
        return image_adjusted_contrast, mask


def adjust_saturation(image, mask, saturation_factor=2):
    """Adjust the saturation of the image by `saturation_factor`
    :param image: either a single image (as tensor) or a tuple, can be sent in batches
    :param mask: the corresponding mask, can be sent in batches, is not transformed
    :param saturation_factor: the facture by which the saturation is changed, should be in range [0, inf], default is 2
    :returns: either transformed (image, image), mask or image, mask
    """
    if isinstance(image, tuple):
        image = image[0]
        two_inputs = True
    else:
        two_inputs = False
    image_adjusted_saturation = tf.image.adjust_saturation(image, saturation_factor=saturation_factor)
    if two_inputs:
        return (image_adjusted_saturation, image_adjusted_saturation), mask
    else:
        return image_adjusted_saturation, mask
