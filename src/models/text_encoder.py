"""
Textual feature extraction using IndicBERT.

Pipeline: Meme Text -> Preprocessing -> Tokenization -> IndicBERT -> Text Embedding
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "ai4bharat/indic-bert"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns the [CLS] embedding representing the whole text sample."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embedding


if __name__ == "__main__":
    # Quick smoke test
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    encoder = TextEncoder("ai4bharat/indic-bert")

    sample = tokenizer(["yeh movie bahut accha tha!"], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = encoder(sample["input_ids"], sample["attention_mask"])
    print("Text embedding shape:", emb.shape)
