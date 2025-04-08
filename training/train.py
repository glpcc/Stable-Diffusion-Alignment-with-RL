import logging
import os
import pathlib

import numpy as np
import torch
import yaml
from clip import get_clip_image_embedding, get_clip_text_embedding
from PIL import Image
from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

logging.basicConfig(level=logging.INFO)


path = pathlib.Path(__file__).parent.resolve()


class ImageSimilarityScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the reference image
    """

    def __init__(self,reference_image, *, dtype):
        super().__init__()
        self.reference_image_features = get_clip_image_embedding(reference_image)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, image):
        embed = get_clip_image_embedding(image)
        return torch.nn.functional.cosine_similarity(embed, self.reference_image_features, dim=-1) *10


class TextSimilarityScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the clip text prompt

    """

    def __init__(self,reference_text, *, dtype):
        super().__init__()
        self.reference_features = get_clip_text_embedding(reference_text)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, image):
        embed = get_clip_image_embedding(image)
        return torch.nn.functional.cosine_similarity(embed, self.reference_features, dim=-1)*10 




def image_outputs_logger(image_data, global_step, accelerate_logger):
    result = {}
    for data in image_data[-4:]:
        images, prompts, _, rewards, _ = data

        for i, image in enumerate(images):
            prompt = prompts[i]
            reward = rewards[i].item()
            result[f"{prompt}__{reward:.5f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )

def load_config(config_path,run_name):
    with open(path / config_path, "r") as f:
        config = yaml.safe_load(f)
    config["ddpo_config"]["run_name"] = run_name
    # Create the folder for the run if it doesn't exist
    run_folder = path / "runs" / run_name
    os.makedirs(run_folder, exist_ok=True)
    # Set the logging and project directories in the config
    config["ddpo_config"]["project_kwargs"]["logging_dir"] = str(run_folder / "logs")
    config["ddpo_config"]["project_kwargs"]["project_dir"] = str(run_folder / "save")

    # copy the config file to the run folder
    config_file_path = run_folder / "config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(config, f)
    

    return config

def load_pipeline(config):
    pipeline = DefaultDDPOStableDiffusionPipeline(
        pretrained_model_name=config["script_config"]["pretrained_model_name"],
        pretrained_model_revision =config["script_config"]["pretrained_revision"],
        use_lora=config["script_config"]["use_lora"]
    )
    if config["resume_from"] != "":
        pipeline.load_lora_weights(config["resume_from"],weight_name="pytorch_lora_weights.safetensors")
    
    return pipeline

def load_scorer(config):
    match config["scorer"]:
        case "text":
            reference_text = config["reference_text"]
            scorer = TextSimilarityScorer(reference_text=reference_text,dtype=torch.float32)
        case "image":
            reference_image = Image.open(config["reference_image"])
            scorer = ImageSimilarityScorer(reference_image=reference_image,dtype=torch.float32)
        case "custom":
            raise NotImplementedError("Custom scorer not implemented.")
        case _:
            raise ValueError("Invalid scorer type. Choose 'text' or 'image'.")
    
    def scorer_fn(images, prompts, model):
        scores = scorer(images)
        print(scores)
        return scores, {}
    
    return scorer_fn

def train(run_name: str):
    config = load_config("train_config.yaml",run_name)
    parser = HfArgumentParser(DDPOConfig)
    ddpo_config, = parser.parse_dict(config["ddpo_config"])
    pipeline = load_pipeline(config)
    scorer_fn = load_scorer(config)
    trainer = DDPOTrainer(
        ddpo_config,
        reward_function=scorer_fn,
        prompt_function=lambda : (np.random.choice(config["train_prompts"]), {}),
        sd_pipeline=pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Create the directory for saving the model
    os.makedirs(path / "runs" / run_name / "final_model", exist_ok=True)
    trainer.save_model(
        save_dir=path / "runs" / run_name / "final_model",
    )
if __name__ == "__main__":
    run_name = input("Enter run name: ")
    train(run_name)