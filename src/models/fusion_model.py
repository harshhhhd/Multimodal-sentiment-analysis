"""
Multimodal fusion model for IndicSent-M.

Supports two fusion strategies (compared per project Section 7):
  - "concat"    : early fusion via concatenation
  - "attention" : cross-modal co-attention fusion
"""

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.image_encoder import ImageEncoder


class CoAttentionFusion(nn.Module):
    """Simple cross-modal attention: text embedding attends to image embedding
    and vice versa, then the attended vectors are concatenated."""

    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.output_dim = hidden_dim * 2

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb).unsqueeze(1)    # (B, 1, H)
        v = self.image_proj(image_emb).unsqueeze(1)  # (B, 1, H)
        seq = torch.cat([t, v], dim=1)                # (B, 2, H)
        attended, _ = self.attn(seq, seq, seq)
        fused = attended.reshape(attended.size(0), -1)  # (B, 2*H)
        return fused


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "ai4bharat/indic-bert",
        image_model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 3,
        fusion_type: str = "concat",  # "concat" | "attention" | "text_only" | "image_only"
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        self.text_encoder = TextEncoder(text_model_name) if fusion_type != "image_only" else None
        self.image_encoder = ImageEncoder(image_model_name) if fusion_type != "text_only" else None

        if fusion_type == "concat":
            fused_dim = self.text_encoder.output_dim + self.image_encoder.output_dim
        elif fusion_type == "attention":
            self.fusion = CoAttentionFusion(self.text_encoder.output_dim, self.image_encoder.output_dim)
            fused_dim = self.fusion.output_dim
        elif fusion_type == "text_only":
            fused_dim = self.text_encoder.output_dim
        elif fusion_type == "image_only":
            fused_dim = self.image_encoder.output_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        if self.fusion_type == "text_only":
            fused = self.text_encoder(input_ids, attention_mask)
        elif self.fusion_type == "image_only":
            fused = self.image_encoder(pixel_values)
        else:
            text_emb = self.text_encoder(input_ids, attention_mask)
            image_emb = self.image_encoder(pixel_values)
            if self.fusion_type == "concat":
                fused = torch.cat([text_emb, image_emb], dim=1)
            else:  # attention
                fused = self.fusion(text_emb, image_emb)

        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # Quick smoke test
    model = MultimodalSentimentModel(fusion_type="concat", num_classes=3)
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    pixel_values = torch.randn(2, 3, 224, 224)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    print("Logits shape:", logits.shape)  # expect (2, 3)
