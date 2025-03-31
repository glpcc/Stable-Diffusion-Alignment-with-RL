import os
import uuid
import re
import pathlib

import pandas as pd
import torch
import yaml
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          QuantoConfig)


# Cargar Configuración de la ejecución
config_file = pathlib.Path(__file__).parent / "bias_detection_config.yaml"
config_file = str(config_file)
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
if config["run_id"] == "":
    run_id = str(uuid.uuid4())
    config["run_id"] = run_id

# Guardar la configuración de la ejecución
os.makedirs("runs/" + run_id, exist_ok=True)
with open(f"runs/{run_id}/config.yaml", 'w') as file:
    yaml.dump(config, file)
# Crear carpeta para las imágenes generadas
os.makedirs("runs/" + run_id + "/images", exist_ok=True)

# Cargar el modelo stable diffusion y el procesador
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# Cargar un modelo preentrenado para medir la calidad
if config["checkpoint"] != "":
    pipe.load_lora_weights(config["checkpoint"], weight_name="pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda")

# Generar imágenes
prompt = config["prompt"]
negative_prompt = config["negative_prompt"]
total_images = config["total_images"]
batch_size = config["batch_size"]
folder = "runs/" + run_id + "/images"

for i in range(0, total_images, batch_size):
    # Saltar si ya se generaron todas las imágenes
    if all(os.path.exists(f"{folder}/image_{i+j}.png") for j in range(batch_size)):
        continue
    images = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        num_images_per_prompt = batch_size,
    )
    for j, image in enumerate(images.images):
        image.save(f"{folder}/image_{i+j}.png")
        print(f"Image {i+j} saved.")

# Borrar el modelo de la GPU
del pipe

# Cargar el modelo Gemma3 y quantizarlo
quantization_config = QuantoConfig(weights="int8")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", device_map="cuda", quantization_config=quantization_config )
final_results = {
    "image": [],
    "race": []
}
# Cargar el regex para filtrar la respuesta
pattern = re.compile(config['respose_regex'], re.IGNORECASE)
# Procesar las imágenes generadas
testing_images_folder = "runs/" + run_id + "/images"
testing_images = os.listdir(testing_images_folder)
for image in testing_images:
    image_path = os.path.join(testing_images_folder, image)
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
    response = re.findall(pattern, generated_texts[0])[0]
    # Guardar la imagen y la respuesta
    final_results["image"].append(image_path)
    final_results["result"].append(response)
    print(response)

# Guardar los resultados en un archivo CSV
df = pd.DataFrame(final_results)
df.to_csv(f"runs/{run_id}/Predicted.csv", index=False)