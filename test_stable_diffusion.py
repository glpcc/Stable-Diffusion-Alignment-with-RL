from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.load_lora_weights("save/checkpoints/checkpoint_12/",weight_name="pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda")

prompt = "photo of a person, highlight hair, sitting outside restaurant, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores"
negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"


# Generate 50 images in batches of 5
total_images = 50
batch_size = 5
folder = "TestGeneration_checkpoint12"
os.makedirs(folder, exist_ok=True)
for i in range(0, total_images, batch_size):
    images = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        num_images_per_prompt = batch_size,
    )
    for j, image in enumerate(images.images):
        image.save(f"{folder}/image_{i+j}.png")

