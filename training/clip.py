from transformers import CLIPModel, CLIPProcessor
import torch


clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip.to(device)

def get_clip_image_embedding(images):
    """
    Get the CLIP image embedding for a given image.

    Args:
        images : Array of images or a single image.

    Returns:
        torch.Tensor: The CLIP image embedding.
    """
    # Load the CLIP model and processor
    
    processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Preprocess the image
    inputs = processor(images=images, return_tensors="pt",do_rescale=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move the inputs to the appropriate device (GPU or CPU)
    inputs = {k: v.to(torch.float32).to(device) for k, v in inputs.items()}
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

    # Preprocess the text
    inputs = processor(text=text, return_tensors="pt")
    inputs.to(device)
    # Get the text features from the model
    text_features = clip.get_text_features(**inputs)

    # Normalize the features
    text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)

    return text_features