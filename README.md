# Semantic Segmentation of Wounds

**Course**: KEN4244 - Deep Learning for Image & Video Processing

**Programm**: Master Data Science for Decision Making

**Academic Year**: 2023/2024

This repository contains all related code and documentation for the project in the course Deep Learning for 
Image & Video Processing at Maastricht University.

## Data
2 data sets are used in this project.

The data from WSNet (available over https://github.com/subbareddy248/WSNET/) should be placed in the directories 
`src/data/wound_images` and `src/data/wound_masks`respectively.

The data set from the Diabetes Foot Ulcer Challenge 2021 (available over https://github.com/uwm-bigdata/wound-segmentation)
should be placed in the folder `src/data/DiabetesFootUlcerChallenge2021` in the same two folders. The data from their 
train and validation set should be combined for testing purposes.

## Project structure

The model architectures can be found in `src/WSNET/models`.

To train all models, the script `train_all_models.py` can be used.

The scripts to load and evaluate already trained models are contained in `src/WSNET/models`
- `evaluate_all_models.py` to evaluate all model architectures with `MobileNet` backbone on the WSNet test set
- `evaluate_densenet.py` to evaluate the models with `densenet121`backbone on the WSNet test set
- `evaluate_dfuc.py` to evaluate the `MobileNet`-backbone models on the data from the Diabetes Foot Ulcer Challenge 2021
- `evaluate_all_models_with_augmentations.py` to evaluate the `MobileNet`-backbone models on the data from WSNet with different augmentations

Results from the evaluations can be found either as `.json` or `.md` files in the directory `src/WSNET/results`.

A notebook containing the results including a visualisation of the WSNet models is also given in 
`src/WSNET_models.ipynb`.