import os
import uuid
import re
import pathlib

import pandas as pd
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image

folder = pathlib.Path(__file__).parent 

def generate_image(prompt):
    run_folder = folder / "generated_images"
    # Crear carpeta para las imagenes generadas
    images_folder = run_folder
    images_folder.mkdir(parents=True, exist_ok=True)

    # Cargar el modelo stable diffusion y el procesador
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


    pipe = pipe.to("cuda")

    # Generar imagenes
    negative_prompt = "blurry, out of focus, low quality, no face visible, painting, cartoon, videogame."
    total_images = 1
    batch_size = 1
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=batch_size,
        num_inference_steps=80,
    ).images[0]
    image_path = images_folder / f"image_{prompt}.png"
    image.save(image_path)
    print(f"Image {prompt} saved.")




if __name__ == "__main__":
    prompts = [
    "Portrait photo of a black person wearing glasses , high quality, detailed, 8k.",
    "Portrait photo of a black person with a hat , high quality, detailed, 8k.",
    "Portrait photo of a black person in a business suit , high quality, detailed, 8k.",
    "Portrait photo of a black person with curly hair , high quality, detailed, 8k.",
    "Portrait photo of a black person wearing a scarf , high quality, detailed, 8k.",
    "Portrait photo of a black person with a beard , high quality, detailed, 8k.",
    "Portrait photo of a black person in a casual outfit , high quality, detailed, 8k.",
    "Portrait photo of a black person with long hair , high quality, detailed, 8k.",
    "Portrait photo of a black person wearing a hoodie , high quality, detailed, 8k.",
    "Portrait photo of a black person with sunglasses , high quality, detailed, 8k.",
    "Portrait photo of a black person, high quality, detailed, 8k.",
    ]   
    for prompt in prompts:
        generate_image(prompt)