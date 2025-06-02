import pathlib
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from PIL import Image

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip.to(device)

def get_clip_image_embedding(image):
    # Load the CLIP model and processor
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move the inputs to the appropriate device (GPU or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Get the image features from the model
    image_features = clip.get_image_features(**inputs)

    # Normalize the features
    image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)

    return image_features

def get_clip_text_embedding(text: str):
    """
    Get the CLIP text embedding for a given text.

    Args:
        text (str): The input text.

    Returns:
        torch.Tensor: The CLIP text embedding.
    """
    # Load the CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Preprocess the text
    inputs = tokenizer(text=text, return_tensors="pt")
    inputs.to(device)
    # Get the text features from the model
    text_features = clip.get_text_features(**inputs)

    # Normalize the features
    text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)

    return text_features


if __name__ == "__main__":
    # Plot the images with the similarity score to the testing test
    test_image_folder = pathlib.Path(__file__).parent.resolve() / "testing_images2"
    test_text = "A glass of water"
    all_images = list(test_image_folder.iterdir())
    all_images = [Image.open(image_path) for image_path in all_images]
    all_images_embeddings = get_clip_image_embedding(all_images)
    text_embedding = get_clip_text_embedding(test_text)
    fig, ax = plt.subplots(1, len(all_images), figsize=(20, 20))
    for i, image in enumerate(all_images_embeddings):
        score = torch.nn.functional.cosine_similarity(image, text_embedding, dim=-1).item()
        ax[i].imshow(all_images[i])
        ax[i].set_title(f"Score: {score:.4f}")
        ax[i].axis('off')
    plt.show()