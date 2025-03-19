from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Ultra-realistic portrait of a Black person, deep brown skin with natural texture, high-detail facial features, realistic lighting, soft shadows, cinematic depth of field, professional photography, 8K resolution, intricate details, DSLR quality, hyperrealism."
negative_prompt = "Unrealistic skin, distorted features, extra fingers, blurry, low resolution, cartoonish, over-sharpened, waxy skin, deformed anatomy, unnatural lighting,black and white"

image = pipe(prompt, negative_prompt=negative_prompt).images[0]
image.save("portrait.png")

