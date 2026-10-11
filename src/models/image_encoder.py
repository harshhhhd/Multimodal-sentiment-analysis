"""
Visual feature extraction using a pretrained Vision Transformer (ViT).

Pipeline: Meme Image -> Preprocessing -> ViT -> Image Embedding
"""

import torch
import torch.nn as nn
from transformers import ViTModel


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.output_dim = self.vit.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns the [CLS] embedding representing the whole image."""
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embedding


if __name__ == "__main__":
    # Quick smoke test with a random tensor standing in for a real image
    encoder = ImageEncoder("google/vit-base-patch16-224")
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        emb = encoder(dummy_image)
    print("Image embedding shape:", emb.shape)
