import itertools
import os
from math import ceil

from src.helper import (
    create_markdown_table_str_for_metrics,
    get_architecture_title,
    get_dfuc_dataset,
    get_segmentation_model_title,
)

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model

if __name__ == "__main__":
    test_images_count = 1010
    batch_size = 16
    segmentation_models = ["unet", "linknet", "pspnet", "fpn"]
    model_architectures = ["local", "global", "global-local"]
    activation_functions = ["sigmoid", "relu"]
    markdown_lines = []
    for model_architecture in model_architectures:
        markdown_lines.append(f"# {get_architecture_title(model_architecture)}")
        markdown_lines.append("")
        for segmentation_model in segmentation_models:
            markdown_lines.append(f"### {get_segmentation_model_title(segmentation_model)}")
            markdown_lines.append("")
            for activation_function in activation_functions:
                markdown_lines.append(f"##### {activation_function.title()}")
                markdown_lines.append("")
                model, _, _, _ = train_model(
                    segmentation_model=segmentation_model,
                    model_architecture=model_architecture,
                    load_only=True,
                    activation_function=activation_function,
                )
                two_inputs = True if model_architecture == "global-local" else False
                dfuc_gen = get_dfuc_dataset(two_inputs=two_inputs)
                results = model.evaluate(dfuc_gen, steps=ceil(float(test_images_count) / float(batch_size)))
                results_dict = dict(zip(model.metrics_names, results))
                markdown_lines.append(create_markdown_table_str_for_metrics(results_dict))
                print(f"Model {segmentation_model} {model_architecture} {activation_function}")
                print(", ".join(f"{key}: {value:.3f}" for key, value in results_dict.items()))
        print("-----")

    activation_function = "sigmoid"
    # mixed models
    combinations = itertools.permutations(segmentation_models, 2)
    markdown_lines.append("# Global-Local models with mixed architectures\n")
    for segmentation_model_tuple in combinations:
        markdown_lines.append(
            f"### Global: {segmentation_model_tuple[0]}, Local: {segmentation_model_tuple[1]}\n"
        )
        model, _, _, _ = train_model(
            segmentation_model=segmentation_model_tuple,
            model_architecture="global-local",
            load_only=True,
            activation_function=activation_function,
        )
        dfuc_gen = get_dfuc_dataset(two_inputs=True)
        results = model.evaluate(dfuc_gen, steps=ceil(float(test_images_count) / float(batch_size)))
        results_dict = dict(zip(model.metrics_names, results))
        markdown_lines.append(create_markdown_table_str_for_metrics(results_dict))

    with open(os.path.join("results", "evaluation_results_dfuc.md"), "w") as writer:
        for line in markdown_lines:
            writer.write(line)
            writer.write("\n")
