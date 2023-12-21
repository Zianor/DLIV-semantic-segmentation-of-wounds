from math import ceil

import segmentation_models as sm
import tensorflow as tf

from src.helper import get_checkpoint_path
from src.WSNET.global_global_model import create_global_global_model
from src.WSNET.global_local_model import create_global_local_model
from src.WSNET.global_local_model_mixed import create_global_local_model_mixed
from src.WSNET.global_model import create_global_model
from src.WSNET.helper import get_datasets, split_train_test_validation
from src.WSNET.local_model import create_local_model


def train_model(
    segmentation_model,
    model_architecture,
    input_size=192,
    load_only=True,
    batch_size=16,
    epochs=100,
    backbone="mobilenet",
    activation_function="sigmoid",
):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet", can be a tuple of (global_model, local_model)
     for "global-local" architecture or (global_model, global_model) for "global-global" architecture
    :param model_architecture: one of "local", "global-local", "global" or "global-global"
    :param input_size: width and height of input images, inputs must be square image. Default is 192. Must be dividable
    by 48 and 64
    :param load_only: if True, the model is loaded from weights, else it is trained
    :param batch_size: batch size, default is 16
    :param epochs: number of epochs, default 100
    :param backbone: name of the backbone that should be used, default is mobilenet
    :param activation_function: activation function, default is sigmoid
    """
    if type(segmentation_model) == tuple and model_architecture not in ["global-local", "global-global"]:
        raise ValueError(
            "Parameter segementation_model must be of type string if the model architecture is local or global"
        )
    if type(segmentation_model) == tuple:
        segmentation_model_str = f"{segmentation_model[0]}-{segmentation_model[1]}"
    else:
        segmentation_model_str = segmentation_model
    if backbone == "mobilenet":
        backbone_str = ""
    else:
        backbone_str = f"-{backbone}"
    if activation_function == "sigmoid":
        checkpoint_name = f"{segmentation_model_str}-{model_architecture}{backbone_str}"
    else:
        checkpoint_name = f"{segmentation_model_str}-{model_architecture}{backbone_str}-{activation_function}"
    two_inputs = False
    if model_architecture == "local":
        model = create_local_model(
            segmentation_model, input_size, backbone=backbone, activation_function=activation_function
        )
    elif model_architecture == "global-local":
        two_inputs = True
        if type(segmentation_model) == str:
            model = create_global_local_model(
                segmentation_model, input_size, backbone=backbone, activation_function=activation_function
            )
        else:
            model = create_global_local_model_mixed(
                segmentation_model_global=segmentation_model[0],
                segmentation_model_local=segmentation_model[1],
                input_size=input_size,
                backbone=backbone,
                activation_function=activation_function,
            )
    elif model_architecture == "global":
        model = create_global_model(
            segmentation_model, input_size, backbone=backbone, activation_function=activation_function
        )
    elif model_architecture == "global-global":
        two_inputs = True
        if not isinstance(segmentation_model, tuple):
            raise ValueError("Parameter segmentation_model must be a tuple for global-global architecture")
        model = create_global_global_model(
            segmentation_model_1=segmentation_model[0],
            segmentation_model_2=segmentation_model[1],
            input_size=input_size,
            backbone=backbone,
            activation_function=activation_function,
        )
    else:
        raise ValueError(
            'Parameter model_architecture must be one of "local", "global-local", "global", "global-global"'
        )

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
