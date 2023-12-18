import os
from math import ceil

os.environ["SM_FRAMEWORK"] = "tf.keras"

from src.WSNET.wsnet import train_model


def get_architecture_title(model_architecture):
    architectures = {"local": "Local model", "global": "Global model", "global-local": "Global-Local model"}
    return architectures[model_architecture]


def get_segmentation_model_title(model):
    models = {"unet": "U-Net", "linknet": "Linknet", "pspnet": "PSPNet", "fpn": "FPN"}
    return models[model]


def create_markdown_table_str_for_metrics(results_dict):
    markdown_lines = []
    table_keys = "| " + " | ".join(f"{key:<5}" for key in results_dict.keys()) + " |"
    markdown_lines.append(table_keys)
    markdown_lines.append("|" + "|".join("-" * max(len(key) + 2, 7) for key in results_dict.keys()) + "|")
    table_values = (
        "| "
        + " | ".join(
            f"{value:.3f}{' ' * (len(key) - 5) if len(key) > 5 else ''}"
            for key, value in results_dict.items()
        )
        + " |"
    )
    markdown_lines.append(table_values)
    markdown_lines.append("")
    return "\n".join(markdown_lines)


if __name__ == "__main__":
    test_images_count = 403
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
                model, train_gen, val_gen, test_gen = train_model(
                    segmentation_model=segmentation_model,
                    model_architecture=model_architecture,
                    load_only=True,
                    activation_function=activation_function,
                )
                results = model.evaluate(test_gen, steps=ceil(float(test_images_count) / float(batch_size)))
                results_dict = dict(zip(model.metrics_names, results))
                markdown_lines.append(create_markdown_table_str_for_metrics(results_dict))
                print(f"Model {segmentation_model} {model_architecture} {activation_function}")
                print(", ".join(f"{key}: {value:.3f}" for key, value in results_dict.items()))
        print("-----")

    with open("evaluation_results.md", "w") as writer:
        for line in markdown_lines:
            writer.write(line)
            writer.write("\n")
