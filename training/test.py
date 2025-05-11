from train import ImageSimilarityScorer
from PIL import Image
import torch
from pathlib import Path

scorer = ImageSimilarityScorer(None, dtype=torch.float32)


path = Path(__file__).parent.resolve()
if __name__ == "__main__":
    test_image = Image.open(path / "ti3.png")
    test_image = test_image.convert("RGB")
    for i in range(10):
        reward = scorer(test_image, None)
        print(reward)