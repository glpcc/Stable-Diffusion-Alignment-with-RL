

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available
from PIL import Image
import json

# Load the CLIP image model
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the CLIP text model
clip_text = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor_text = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the image
image = Image.open("reference_image.jpg")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")
image_features = clip.get_image_features(**inputs)
image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)

# Create text embedding
prompt = "a black person"
inputs_text = processor_text(text=prompt, return_tensors="pt")
text_features = clip_text.get_text_features(**inputs_text)
text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)

# Save the image features
resultd = dict()
resultd['image_features'] = image_features.tolist()
resultd['text_features'] = text_features.tolist()
with open('features.json', 'w') as f:
    json.dump(resultd, f)