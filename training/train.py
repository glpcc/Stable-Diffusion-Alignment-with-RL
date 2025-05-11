import logging
import os
import pathlib
import random

import numpy as np
import torch
import yaml
from clip import get_clip_image_embedding, get_clip_text_embedding
from PIL import Image
from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)


path = pathlib.Path(__file__).parent.resolve()


class ImageSimilarityScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the reference image
    """

    def __init__(self,config, *, dtype):
        super().__init__()
        reference_image_path = path / "reference_images"
        self.reference_images_features = []
        self.reference_images = reference_image_path.glob("*")
        for image_path in self.reference_images:
            image = Image.open(image_path).convert("RGB")
            emb = get_clip_image_embedding(image,do_rescale=True)
            print(f"Image path: {image_path}, embedding shape: {emb.shape}")
            self.reference_images_features.append(emb)
        
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, images, prompts):
        embed = get_clip_image_embedding(images)
        random_references = random.sample(self.reference_images_features,1)[0] 
        print(random_references.shape)
        result = torch.nn.functional.cosine_similarity(embed, random_references, dim=-1)*10
        return result


class TextSimilarityScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the clip text prompt

    """

    def __init__(self,reference_text, *, dtype):
        super().__init__()
        self.reference_features = get_clip_text_embedding(reference_text)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, image, prompts):
        embed = get_clip_image_embedding(image)
        return torch.nn.functional.cosine_similarity(embed, self.reference_features, dim=-1)*10 

class RandomTextSimilarityScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the clip text prompt

    """

    def __init__(self,reference_text1,reference_text2, *, dtype):
        super().__init__()
        self.reference_features1 = get_clip_text_embedding(reference_text1)
        self.reference_features2 = get_clip_text_embedding(reference_text2)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, image, prompts):
        embed = get_clip_image_embedding(image)
        if random.random() > 0.5:
            return torch.nn.functional.cosine_similarity(embed, self.reference_features1, dim=-1)*10
        else:
            return torch.nn.functional.cosine_similarity(embed, self.reference_features2, dim=-1)*10

class DisparityScoreScorer():
    """
    This model attempts to make the generated images to be as similar as possible to the clip text prompt

    """

    def __init__(self,reference_text1,reference_text2, *, dtype):
        super().__init__()
        print(f"Reference text 1: {reference_text1}, reference text 2: {reference_text2}")
        self.reference_features1 = get_clip_text_embedding(reference_text1)
        self.reference_features2 = get_clip_text_embedding(reference_text2)
        # Asume the reference 2 is the majority
        self.last_20_generations = {
            "Mugshot" : [2]*10,
            "Portrait photo of a security guard, high quality, detailed, 8k.": [2]*10,
            "Portrait photo of a person, high quality, detailed, 8k.": [2]*10,
            "Portrait photo of a police officer, high quality, detailed, 8k.": [2]*10,
            "Portrait photo of a nurse, high quality, detailed, 8k.": [1]*10,
            "Portrait photo of a doctor, high quality, detailed, 8k.": [2]*10,
        }
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, image, prompts):
        embed = get_clip_image_embedding(image)
        similarity_with_one = torch.nn.functional.cosine_similarity(embed, self.reference_features1, dim=-1)*5
        similarity_with_two = torch.nn.functional.cosine_similarity(embed, self.reference_features2, dim=-1)*5
        print(f"Similarity with one: {similarity_with_one}, similarity with two: {similarity_with_two}")
        p = prompts[0]
        self.last_20_generations[p].pop(0)
        if similarity_with_two > similarity_with_one:
            self.last_20_generations[p].append(2)
        else:
            self.last_20_generations[p].append(1)
        print(self.last_20_generations[p])
        two_count = self.last_20_generations[p].count(2)
        one_count = len(self.last_20_generations[p]) - two_count
        print(f"Two count: {two_count}, One count: {one_count}")
        if two_count >= one_count:
            id_score = one_count / two_count
            print(f"ID score: {id_score}")
            return similarity_with_one + id_score
        else:
            id_score = two_count / one_count
            print(f"ID score: {id_score}")
            return similarity_with_two + id_score
        
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
            print(f"You are using the image similarity scorer")
            scorer = ImageSimilarityScorer(config,dtype=torch.float32)
        case "random":
            reference_text1 = config["reference_text1"]
            reference_text2 = config["reference_text2"]
            scorer = RandomTextSimilarityScorer(reference_text1=reference_text1,reference_text2=reference_text2,dtype=torch.float32)
        case "idscore":
            reference_text1 = config["reference_text1"]
            reference_text2 = config["reference_text2"]
            scorer = DisparityScoreScorer(reference_text1=reference_text1,reference_text2=reference_text2,dtype=torch.float32)
        case _:
            raise ValueError("Invalid scorer type. Choose 'text' or 'image'.")
    
    def scorer_fn(images, prompts, model):
        scores = scorer(images,prompts)
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
    trainer.sd_pipeline.save_pretrained(path / "runs" / run_name / "final_model")

if __name__ == "__main__":
    run_name = input("Enter run name: ")
    train(run_name)