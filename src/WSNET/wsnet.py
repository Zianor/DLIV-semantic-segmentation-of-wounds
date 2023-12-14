from math import ceil

import segmentation_models as sm
import tensorflow as tf

from src.helper import get_checkpoint_path
from src.WSNET.global_local_model import create_global_local_model
from src.WSNET.helper import get_datasets, split_train_test_validation
from src.WSNET.local_model import create_local_model
from WSNET.global_model import create_global_model


def train_model(
    segmentation_model, model_architecture, input_size=192, load_only=True, batch_size=16, epochs=100
):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet"
    :param model_architecture: one of "local", "global-local"
    :param input_size: width and height of input images, inputs must be square image. Default is 192. Must be dividable
    by 48 and 64
    :param load_only: if True, the model is loaded from weights, else it is trained
    :param batch_size: batch size, default is 16
    :param epochs: number of epochs, default 100
    """
    checkpoint_name = f"{segmentation_model}-{model_architecture}"
    two_inputs = False
    if model_architecture == "local":
        model = create_local_model(segmentation_model, input_size)
    elif model_architecture == "global-local":
        model = create_global_local_model(segmentation_model, input_size)
        two_inputs = True
    elif model_architecture == "global":
        model = create_global_model(segmentation_model, input_size)
    else:
        raise ValueError('Parameter model_architecture must be one of "local", "global-local"')

    checkpoint_path = get_checkpoint_path(checkpoint_name, False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, save_best_only=True, mode="min"
        )
    ]

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        "Adam",
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), "binary_accuracy"],
    )

    train_images, validation_images, test_images = split_train_test_validation()

    train_gen, val_gen, test_gen = get_datasets(
        train_images,
        validation_images,
        test_images,
        two_inputs=two_inputs,
        input_size=input_size,
        batch_size=batch_size,
    )

    if not load_only:
        model.fit(
            train_gen,
            steps_per_epoch=ceil(float(len(train_images)) / float(batch_size)),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=ceil(float(len(validation_images)) / float(batch_size)),
            verbose=1,
        )
    else:
        model.load_weights(checkpoint_path)

    return model, train_gen, val_gen, test_gen
