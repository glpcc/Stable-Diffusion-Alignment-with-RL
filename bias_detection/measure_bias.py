import os
import uuid
import re
import pathlib

import pandas as pd
import torch
import yaml
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          QuantoConfig)

folder = pathlib.Path(__file__).parent 
def load_config():
    # Cargar Configuracion de la ejecucion
    config_file = pathlib.Path(__file__).parent / "bias_detection_config.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config):
    if config["run_id"] == "":
        run_id = str(uuid.uuid4())
        config["run_id"] = run_id
    else:
        run_id = config["run_id"]
    # Guardar la configuracion de la ejecucion
    run_folder = folder / "runs" / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    config_path = run_folder / "config.yaml"

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def generate_images(config):
    run_folder = folder / "runs" / config["run_id"]
    # Crear carpeta para las imagenes generadas
    images_folder = run_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # Cargar el modelo stable diffusion y el procesador
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Cargar un modelo preentrenado para medir la calidad
    if config["checkpoint"] != "" and "checkpoint_0" not in config["checkpoint"]:
        print("Loading checkpoint")
        pipe.load_lora_weights(config["checkpoint"], weight_name="pytorch_lora_weights.safetensors")


    pipe = pipe.to("cuda")

    # Generar imagenes
    prompt = config["prompt_sd"]
    negative_prompt = config["negative_prompt_sd"]
    total_images = config["num_images"]
    batch_size = config["sd_batch_size"]

    for i in range(0, total_images, batch_size):
        # Saltar si ya se generaron todas las imagenes
        if all((images_folder / f"image_{i+j}.png").exists() for j in range(batch_size)):
            continue
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=config["sd_steps"],
        )
        for j, image in enumerate(images.images):
            image_path = images_folder / f"image_{i+j}.png"
            image.save(image_path)
            print(f"Image {i+j} saved.")

    # Borrar el modelo de la GPU
    del pipe

def generate_llm_predictions(config):
    run_folder = folder / "runs" / config["run_id"]
    # Cargar el modelo Gemma3 y quantizarlo
    quantization_config = QuantoConfig(weights="int8")
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it",use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", device_map="cuda", quantization_config=quantization_config)
    final_results = {
        "image": [],
        "race": [],
        "gender": [],
    }



    # Cargar el regex para filtrar la respuesta
    pattern_race = re.compile(config['respose_regex_race'], re.IGNORECASE)
    pattern_gender = re.compile(config['respose_regex_gender'], re.IGNORECASE)
    # Procesar las imagenes generadas
    images_folder = run_folder / "images"
    testing_images = list(images_folder.iterdir())
    for image_path in testing_images:

        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": config["prompt_gemma"]},
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
        # Generar la respuesta
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=500)
        # Decodificar la respuesta
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # Obtener la respuesta
        race = re.findall(pattern_race, generated_texts[0])
        gender = re.findall(pattern_gender, generated_texts[0])
        if len(race) == 0:
            race = "None"
        else:
            race = race[0]
        if len(gender) == 0:
            gender = "None"
        else:
            gender = gender[0]
        # Guardar la imagen y la respuesta
        final_results["image"].append(str(image_path))
        final_results["race"].append(race)
        final_results["gender"].append(gender)
        print(f"Processed image {image_path}")
        
    # Guardar los resultados en un archivo CSV
    results_path = run_folder / "Predicted.csv"
    df = pd.DataFrame(final_results)
    df.to_csv(results_path, index=False)

if __name__ == "__main__":
    # Cargar la configuracion
    config = load_config()
    # Guardar la configuracion
    save_config(config)
    # Generar las imagenes
    generate_images(config)
    # Generar las predicciones
    generate_llm_predictions(config)