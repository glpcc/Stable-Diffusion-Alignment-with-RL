import os
import uuid
import re
import pathlib

import pandas as pd
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image

folder = pathlib.Path(__file__).parent 
# Cargar el modelo stable diffusion y el procesador
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt, run_id=""):
    run_folder = folder / "generated_images"
    # Crear carpeta para las imagenes generadas
    images_folder = run_folder
    images_folder.mkdir(parents=True, exist_ok=True)

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
    image_path = images_folder / f"image_{run_id}_{prompt}.png"
    image.save(image_path)
    print(f"Image {prompt} saved.")




if __name__ == "__main__":
    prompts = [
    "Portrait photo of a business executive, high quality, detailed, 8k.",
    ]   
    num_images = 10
    for i in range(num_images):
        for prompt in prompts:
            generate_image(prompt,i)