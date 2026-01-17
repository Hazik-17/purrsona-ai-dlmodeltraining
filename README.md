# Cat Breed Detection - Model Training Project

## What This Project Does
This repository contains the model training pipeline for a cat-breed detection system intended for a mobile app. It trains multiple transfer-learning classifiers (binary gatekeeper + breed classifiers) and exports them for mobile deployment (Keras `.keras` exports and TensorFlow Lite `.tflite` conversions).

## Models Trained (overview)
- **Gatekeeper (cat_vs_not_cat.keras)**
  - Purpose: Binary classifier to detect whether an image contains a cat (used as first-stage filter).
  - Type: EfficientNetV2B0 (transfer learning)
  - Classes: 2 (cat / not_cat)
  - Accuracy: reported high (>99% in conversion comments); see experimental logs for runs.
  - Output File: `cat_vs_not_cat.keras`

- **Generalist / Breed Expert (breed_expert_efficientnetv2b0_v2.keras)**
  - Purpose: Classify into the main set of cat breeds (used as the main breed predictor).
  - Type: EfficientNetV2B0 (transfer learning)
  - Classes: 12 cat breeds
  - Accuracy: example run logged at ~82.95% test accuracy (see experiment logs); conversion comments reference ~93.47% for best run.
  - Output File: `breed_expert_efficientnetv2b0_v2.keras`

- **Similar-Breed Expert (similar_breed_expert_effnet_v1.keras)**
  - Purpose: Specialized classifier for a subset of visually-similar breeds (refinement stage).
  - Type: EfficientNetV2B0 (transfer learning)
  - Classes: 5 (Birman, British, Maine, Persian, Ragdoll)
  - Accuracy: conversion comments reference ~95.50% for a top run.
  - Output File: `similar_breed_expert_effnet_v1.keras`

- **Other variants present in repo**: ResNet50 and MobileNetV3 variants are included (for experimentation). Typical filenames: `breed_expert_resnet50_v1.keras`, `breed_expert_mobilenetV3.keras`.

## Dataset Information

### Primary Dataset
- Name: Oxford-IIIT Pet Dataset (processed and organized for this project)
- Source: original dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Repo copy / processed labels: [preprocessingDataset/output/labels.csv](preprocessingDataset/output/labels.csv)
- Total images (processed / augmented listing): ~6,164 entries in `labels.csv` (includes augmented images).
- Classes: 12 cat breeds (Abyssinian, Bengal, Birman, Bombay, British, Egyptian, Maine, Persian, Ragdoll, Russian, Siamese, Sphynx).
- Split: Training and test folders exist in [preprocessingDataset/output/train](preprocessingDataset/output/train) and [preprocessingDataset/output/test](preprocessingDataset/output/test). Training uses an internal 80/20 train/validation split (see `validation_split=0.2` in training scripts).

### Secondary Datasets
- Similar-breed dataset: [preprocessingDataset/similar_breed_dataset](preprocessingDataset/similar_breed_dataset) — contains a 5-class train/test split for fine-grained expert training.
- Binary (cat vs not-cat) dataset: [preprocessingDataset/cat_vs_not_cat_dataset](preprocessingDataset/cat_vs_not_cat_dataset) with `train/cat`, `train/not_cat`, and `test/` folders.

- Other-animal dataset (source for `not_cat` class): A public Kaggle dataset with 90 different animals which is used to build the `not_cat` class when preparing the binary dataset. Download from: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

  Note: Set the `NOT_CAT_SOURCE_DIR` variable in `preprocessingDataset/gpu_train_model/binary_dataset_preparation.py` (or copy the downloaded folders into `preprocessingDataset/others_animal`) before running the binary dataset preparation script.

## Training Process
- Framework: TensorFlow / Keras (tf.keras). See examples: [preprocessingDataset/gpu_train_model/efficientnetV2B0/train_breed_expert_efficientnetv2b0.py](preprocessingDataset/gpu_train_model/efficientnetV2B0/train_breed_expert_efficientnetv2b0.py), [preprocessingDataset/gpu_train_model/resnet50/train_breed_expert_resnet_v1.py](preprocessingDataset/gpu_train_model/resnet50/train_breed_expert_resnet_v1.py), [preprocessingDataset/gpu_train_model/mobilenet_v3/large/train_breed_expert.py](preprocessingDataset/gpu_train_model/mobilenet_v3/large/train_breed_expert.py).
- Image size: 224×224 (consistent across scripts).
- Preprocessing: model-specific `preprocess_input` from each architecture (EfficientNetV2, ResNet, MobileNetV3).
- Data augmentation (examples): random flip (horizontal/vertical), random crop, resize, random brightness, rotation, zoom, width/height shift, shear, brightness range; implemented via tf.data pipeline or `ImageDataGenerator` depending on script.
- Training configuration (typical): two-phase schedule — Phase 1 train top layers only, Phase 2 fine-tune last ~50 base-model layers.
  - Typical epochs: Phase 1 = 20; Phase 2 = 30 (some runs use 25 / 50 for expert models)
  - Typical batch size: 32 (MobileNet variants sometimes use 64)
  - Learning rates: Phase 1 ~1e-3; Phase 2 ~8e-6
  - Callbacks: `EarlyStopping`, `ReduceLROnPlateau`.
- Transfer learning: yes — base models initialized with `weights='imagenet'` and fine-tuned later.
- Base models used for transfer learning: EfficientNetV2B0, ResNet50, MobileNetV3Small.

## Model Evaluation
- Metrics tracked: `accuracy`, training/validation `loss`. Detailed per-class metrics produced via `sklearn.metrics.classification_report` (precision, recall, f1-score).
- Evaluation approach: final `model.evaluate()` on the test dataset and `model.predict()` combined with `classification_report` to produce per-class statistics.
- Example logged run: see [experiment_training_results/text.txt](experiment_training_results/text.txt) — a run shows Test accuracy 82.95% (and training accuracy ~95.83%) with full per-class report.

## Outputs and Artifacts
- Trained Keras models: `.keras` files saved next to training scripts (e.g. `breed_expert_efficientnetv2b0_v2.keras`, `cat_vs_not_cat.keras`, `similar_breed_expert_effnet_v1.keras`).
- Conversion to TensorFlow Lite: scripts in [preprocessingDataset/gpu_train_model/efficientnetV2B0/best_model/model_conversion.py](preprocessingDataset/gpu_train_model/efficientnetV2B0/best_model/model_conversion.py) produce `.tflite` files (used for mobile deployment). Default mapping and example outputs are configured there.
- Class indices: JSON files saved by training scripts (e.g. `breed_expert_class_indices_effnet.json`, `non_cat_class_indices.json`, `similar_breed_class_indices.json`).
- Visualizations: training history plots saved as PNGs (e.g. `phase_1_training_plot_effnet.png`, `phase_2_training_plot_effnet.png`, `phase_1_training_plot.png`, etc.).

## Important Files (jump to)
- Training scripts (EfficientNet): [preprocessingDataset/gpu_train_model/efficientnetV2B0](preprocessingDataset/gpu_train_model/efficientnetV2B0)
- Conversion script: [preprocessingDataset/gpu_train_model/efficientnetV2B0/best_model/model_conversion.py](preprocessingDataset/gpu_train_model/efficientnetV2B0/best_model/model_conversion.py)
- General data outputs: [preprocessingDataset/output/labels.csv](preprocessingDataset/output/labels.csv)

## How to reproduce training (high level)
1. Prepare data under `preprocessingDataset/output/train` and `preprocessingDataset/output/test` (or use the provided processed folders).
2. Activate a Python environment with TensorFlow (2.x), scikit-learn, matplotlib, and dependencies.
3. Run a training script, for example:

```bash
python preprocessingDataset/gpu_train_model/efficientnetV2B0/train_breed_expert_efficientnetv2b0.py
```

4. (Optional) Convert the saved `.keras` models to `.tflite` with:

```bash
python preprocessingDataset/gpu_train_model/efficientnetV2B0/best_model/model_conversion.py
```

## Notes & Next Steps
- The repo contains multiple experimental training scripts and older variants — use the `efficientnetV2B0` folder for the up-to-date pipeline used for mobile targets.
- If you want, I can:
  - Add a minimal `requirements.txt` / `environment.yml` for reproducibility.
  - Add a short script that lists model test accuracies automatically from `experiment_training_results` and conversion comments.

---
Generated by repository analysis. For quick checks, open the main training script: [preprocessingDataset/gpu_train_model/efficientnetV2B0/train_breed_expert_efficientnetv2b0.py](preprocessingDataset/gpu_train_model/efficientnetV2B0/train_breed_expert_efficientnetv2b0.py)
