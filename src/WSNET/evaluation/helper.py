import os
from math import ceil

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model


def evaluate_model(
    model_architecture, segmentation_model, activation_function="sigmoid", backbone="mobilenet"
):
    """
    :param segmentation_model: one of "fpn", "pspnet", "linknet", "unet", can be a tuple of (global_model, local_model)
     for "global-local" architecture
    :param model_architecture: one of "local", "global-local"
    :param backbone: name of the backbone that should be used, default is mobilenet
    :param activation_function: activation function, default is sigmoid
    """
    from WSNET.helper import get_image_counts

    _, _, test_images_count = get_image_counts()
    test_images_count = 403
    batch_size = 16
    model, train_gen, val_gen, test_gen = train_model(
        segmentation_model=segmentation_model,
        model_architecture=model_architecture,
        load_only=True,
        activation_function=activation_function,
        backbone=backbone,
    )
    results = model.evaluate(test_gen, steps=ceil(float(test_images_count) / float(batch_size)))
    results_dict = dict(zip(model.metrics_names, results))
    return results_dict
