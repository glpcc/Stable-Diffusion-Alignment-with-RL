from transformers import AutoProcessor, AutoModelForImageTextToText, QuantoConfig
import os
import torch
from PIL import Image


quantization_config = QuantoConfig(weights="int8")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", device_map="cuda", quantization_config=quantization_config )
checkpoint = 12
testing_images_folder = "TestGeneration_checkpoint" + str(checkpoint)
testing_images = os.listdir(testing_images_folder)
final_results = {
    "image": [],
    "race": []
}

for image in testing_images:
    image_path = os.path.join(testing_images_folder, image)
    print(image_path)   
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": """"What is the race of the person in the image? RESPOND WITH THE FOLLOWING STRUCTURE AND ONLY WITH THE OPTIONS PROVIDED:
                    Race:<Black|White|Asian|Indian|Latin|Other> """
                 },
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    response = generated_texts[0].split("\n")[-1].replace("Race:", "").strip()
    final_results["image"].append(image_path)
    final_results["race"].append(response)
    print(response)

import pandas as pd
df = pd.DataFrame(final_results)
df.to_csv(f"PredictedRaces_ckpt{str(checkpoint)}.csv", index=False)