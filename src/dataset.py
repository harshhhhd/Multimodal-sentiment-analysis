"""
PyTorch Dataset for IndicSent-M.

Loads preprocessed text + image pairs and returns tokenized text tensors,
image tensors (if available) and the sentiment label.
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.preprocessing.image_preprocess import load_image


class MemeSentimentDataset(Dataset):
    def __init__(self, dataframe, config, split="train"):
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.split = split

        self.text_column = config["data"]["text_column"]
        self.image_column = config["data"]["image_column"]
        self.label_column = config["data"]["label_column"]
        self.image_dir = config["data"]["image_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(config["text_model"]["name"])
        self.max_length = config["text_model"]["max_length"]
        self.image_size = config["image_model"]["image_size"]

        self.label_map = {label: i for i, label in enumerate(config["labels"]["classes"])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- text ---
        text = str(row[self.text_column]) if self.text_column in row and row[self.text_column] else ""
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # --- image ---
        image_tensor = torch.zeros(3, self.image_size, self.image_size)
        has_image = torch.tensor(0.0)
        if self.image_column in row and row[self.image_column]:
            image_path = os.path.join(self.image_dir, str(row[self.image_column]))
            img = load_image(image_path, image_size=self.image_size, train=(self.split == "train"))
            if img is not None:
                image_tensor = img
                has_image = torch.tensor(1.0)

        # --- label ---
        raw_label = str(row[self.label_column]).strip().lower()
        label = self.label_map.get(raw_label, 1)  # defaults to 'neutral' index if unseen

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "has_image": has_image,
            "label": torch.tensor(label, dtype=torch.long),
        }
