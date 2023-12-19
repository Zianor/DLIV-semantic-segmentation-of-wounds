import itertools
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model

if __name__ == "__main__":
    segmentation_models = ["unet", "pspnet", "fpn", "linknet"]
    model_architectures = ["global", "global-local", "local"]
    activation_function = "sigmoid"
    backbone = "mobilenet"
    for model_architecture in model_architectures:
        for segmentation_model in segmentation_models:
            train_model(
                segmentation_model=segmentation_model,
                model_architecture=model_architecture,
                load_only=False,
                backbone=backbone,
                activation_function=activation_function,
            )

    # mixed models
    combinations = itertools.permutations(segmentation_models, 2)
    for segmentation_model_tuple in combinations:
        train_model(
            segmentation_model=segmentation_model_tuple,
            model_architecture="global-local",
            load_only=False,
            backbone=backbone,
            activation_function=activation_function,
        )
