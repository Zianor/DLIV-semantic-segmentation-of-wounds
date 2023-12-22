import json
import os
from math import ceil
from pathlib import Path

from src.WSNET.evaluation.augmentations_for_testing import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    embed_in_black_background,
)

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model


def get_results_with_augmentations():
    test_images_count = 403
    batch_size = 16
    segmentation_models = ["unet", "linknet", "pspnet", "fpn"]
    model_architectures = ["local", "global", "global-local"]
    activation_functions = ["sigmoid", "relu"]
    results = {}
    for model_architecture in model_architectures:
        architecture_results = {}
        for segmentation_model in segmentation_models:
            model_results = {}
            for activation_function in activation_functions:
                model, train_gen, val_gen, test_gen = train_model(
                    segmentation_model=segmentation_model,
                    model_architecture=model_architecture,
                    load_only=True,
                    activation_function=activation_function,
                )
                # results for "normal" data
                results_normal = model.evaluate(
                    test_gen, steps=ceil(float(test_images_count) / float(batch_size))
                )
                results_dict_normal = dict(zip(model.metrics_names, results_normal))
                # results for data embedded in black background
                test_gen_embed = test_gen.map(embed_in_black_background)
                results_embed = model.evaluate(
                    test_gen_embed, steps=ceil(float(test_images_count) / float(batch_size))
                )
                results_dict_embed = dict(zip(model.metrics_names, results_embed))
                # results for adjusted brightness
                test_gen_brightness = test_gen.map(adjust_brightness)
                results_brightness = model.evaluate(
                    test_gen_brightness, steps=ceil(float(test_images_count) / float(batch_size))
                )
                results_dict_brightness = dict(zip(model.metrics_names, results_brightness))
                # results for adjusted contrast
                test_gen_contrast = test_gen.map(adjust_contrast)
                results_contrast = model.evaluate(
                    test_gen_contrast, steps=ceil(float(test_images_count) / float(batch_size))
                )
                results_dict_contrast = dict(zip(model.metrics_names, results_contrast))
                # results for adjusted saturation
                test_gen_saturation = test_gen.map(adjust_saturation)
                results_saturation = model.evaluate(
                    test_gen_saturation, steps=ceil(float(test_images_count) / float(batch_size))
                )
                results_dict_saturation = dict(zip(model.metrics_names, results_saturation))
                model_results[activation_function] = {
                    "normal": results_dict_normal,
                    "embed": results_dict_embed,
                    "brightness": results_dict_brightness,
                    "contrast": results_dict_contrast,
                    "saturation": results_dict_saturation,
                }
            architecture_results[segmentation_model] = model_results
        results[model_architecture] = architecture_results
        return results


def write_results_as_json(results_dict):
    with open(os.path.join(Path.cwd().parent, "results", "evaluation_augmentations.json"), "w") as write_file:
        json.dump(results_dict, write_file)


if __name__ == "__main__":
    results = get_results_with_augmentations()
    write_results_as_json(results)
