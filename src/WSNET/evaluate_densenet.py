from src.helper import (
    create_markdown_table_str_for_metrics,
    get_architecture_title,
    get_segmentation_model_title,
)
from WSNET.evaluation import evaluate_model

if __name__ == "__main__":
    test_images_count = 403
    batch_size = 16
    segmentation_models = ["unet", "linknet", "pspnet", "fpn"]
    model_architectures = ["local", "global", "global-local"]
    activation_functions = ["sigmoid"]
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
                results_dict = evaluate_model(
                    segmentation_model=segmentation_model,
                    model_architecture=model_architecture,
                    activation_function=activation_function,
                    backbone="densenet121",
                )
                markdown_lines.append(create_markdown_table_str_for_metrics(results_dict))
                print(f"Model {segmentation_model} {model_architecture} {activation_function}")
                print(", ".join(f"{key}: {value:.3f}" for key, value in results_dict.items()))
        print("-----")
        with open("evaluation_results_densenet.md", "w") as writer:
            for line in markdown_lines:
                writer.write(line)
                writer.write("\n")
