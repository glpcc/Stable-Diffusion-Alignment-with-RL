# Bias reduction in Stable Diffusion 1.5 with Reinforcement Learning

This project is part of a bachelors thesis on the reduction of gender and race bias in generative models, specially using Stable Diffusion 1.5

## Project Structure

### Main Directories

#### `bias_detection/`
- Tools and scripts for detecting and measuring bias in trained models.
- Includes configuration files, bias measurement scripts, and LaTeX report generation.
- `runs/`: Directory that contains the bias test for each checkpoint of different training runs
  - Inside each folder there is a config with the evaluation parameters as well as a csv with the summarized results.

#### `embedding_testing/`
- Scripts for testing and comparing image and text embeddings.
- Contains tests and sample images for embedding evaluation.

#### `training/`
- Core training scripts and configuration files.
- `train.py`: Main training script supporting different reward/scoring functions (text, image, random, disparity).
- `clip.py`: Utilities for extracting CLIP embeddings from images and text.
- `train_config.yaml`: Default configuration for training runs.
- `runs/`: Contains subfolders for each training run that was used, with their own configs and outputs.


## How to Use

1. **Install dependencies:**  
   ```
   pip install -r requirements.txt
   ```

2. **Run training:**  
   You need to configure the training run parameters in the `train_config.yaml` file.
   ```
   python training/train.py
   ```
   You will be prompted to enter a run name, which corresponds to a folder in `training/runs/`.

3. **Bias Detection:**  
   Use scripts in `bias_detection/` to analyze model checkpoints for bias.


## Disclaimer
This is in no way, shape or form a polished module, it is just the code I used during my thesis, it is not well documented, use at your own risk.