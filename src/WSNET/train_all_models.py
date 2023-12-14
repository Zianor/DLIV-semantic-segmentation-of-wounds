import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model

if __name__ == "__main__":
    segmentation_models = ["unet", "pspnet", "fpn", "linknet"]
    model_architectures = ["global", "global-local", "local"]
    for model_architecture in model_architectures:
        for segmentation_model in segmentation_models:
            train_model(
                segmentation_model=segmentation_model, model_architecture=model_architecture, load_only=False
            )
