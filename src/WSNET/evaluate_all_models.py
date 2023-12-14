import os
from math import ceil

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model

if __name__ == "__main__":
    test_images_count = 403
    batch_size = 16
    segmentation_models = ["unet", "pspnet", "fpn", "linknet"]
    model_architectures = ["global-local", "local"]
    for model_architecture in model_architectures:
        for segmentation_model in segmentation_models:
            model, train_gen, val_gen, test_gen = train_model(
                segmentation_model=segmentation_model, model_architecture=model_architecture, load_only=True
            )
            results = model.evaluate(test_gen, steps=ceil(float(test_images_count) / float(batch_size)))
            results_dict = dict(zip(model.metrics_names, results))
            print(f"Model {segmentation_model} {model_architecture}")
            print(", ".join(f"{key}: {value:.3f}" for key, value in results_dict.items()))
            print("--")
        print("-----")
