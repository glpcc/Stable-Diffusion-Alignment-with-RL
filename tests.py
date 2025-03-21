
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python examples/scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available
from PIL import Image
import json
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
    """

    pretrained_model: str = field(
        default="stable-diffusion-v1-5/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class ImageSimilarityScorer(torch.nn.Module):
    """
    This model attempts to make the generated images to be as similar as possible to the reference image, which is a black person."
    """

    def __init__(self, *, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # Calculate reference image features
        reference_image = Image.open("reference_image.jpg")
        inputs = self.processor(images=reference_image, return_tensors="pt")
        inputs = {k: v.to(dtype) for k, v in inputs.items()}
        self.reference_image_features = self.clip.get_image_features(**inputs)
        self.reference_image_features = self.reference_image_features / torch.linalg.vector_norm(
            self.reference_image_features, dim=-1, keepdim=True
        )
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt",do_rescale=False)
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(embed, self.reference_image_features, dim=-1)


class TextSimilarityScorer(torch.nn.Module):
    """
    This model attempts to make the generated images to be as similar as possible to the clip text prompt "a portrait photo of an old black man in a suit""
    """

    def __init__(self, *, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        with open('features.json', 'r') as f:
            features = json.load(f)
            temp = torch.tensor(features['text_features'])
        self.reference_features = temp
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt",do_rescale=False)
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(embed, self.reference_features, dim=-1)


similarity_scorer = TextSimilarityScorer(
    dtype=torch.float32)
# similarity_scorer = ImageSimilarityScorer(
#     dtype=torch.float32)
def scorer(images, prompts, model):
    scores = similarity_scorer(images)*10
    print(scores)
    return scores, {}



# list of example prompts to feed stable diffusion
prompts = [
    "A portrait photo of a person wearing glasses",
    "A ultra realistic portrait of a person with a hat",
    "A portrait photo of a person in a business suit",
    "A ultra realistic portrait of a person with curly hair",
    "A portrait photo of a person wearing a scarf",
    "A ultra realistic portrait of a person with a beard",
    "A face photo of a person in a casual outfit",
    "A ultra realistic portrait of a person with long hair",
    "A portrait photo of a person wearing a hoodie",
    "A ultra realistic portrait of a person with sunglasses",
]


def prompt_fn():
    return np.random.choice(prompts), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
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


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 500,
        "project_dir": "./save",
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    pipeline.sd_pipeline.load_lora_weights(
                "save2/checkpoints/checkpoint_17/",
                weight_name="pytorch_lora_weights.safetensors",
            )
    trainer = DDPOTrainer(
        training_args,
        scorer,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
