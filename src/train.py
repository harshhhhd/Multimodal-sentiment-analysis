"""
Training script for IndicSent-M.

Usage:
    python -m src.train --fusion_type text_only
    python -m src.train --fusion_type concat
    python -m src.train --fusion_type attention
    python -m src.train --fusion_type image_only
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd

from src.dataset import MemeSentimentDataset
from src.models.fusion_model import MultimodalSentimentModel
from src.preprocessing.text_preprocess import preprocess_dataframe


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_splits(config):
    df = pd.read_csv(config["data"]["raw_csv"])
    df = preprocess_dataframe(df, config["data"]["text_column"], config["data"]["image_column"])
    
    label_col = config["data"]["label_column"]
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df = df[df[label_col].isin(config["labels"]["classes"])].reset_index(drop=True)
    print(f"Using {len(df)} labeled rows out of total dataset")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - config["data"]["train_split"]),
        random_state=config["data"]["random_seed"],
        stratify=df[config["data"]["label_column"]] if config["data"]["label_column"] in df else None,
    )
    relative_val = config["data"]["val_split"] / (config["data"]["val_split"] + config["data"]["test_split"])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val),
        random_state=config["data"]["random_seed"],
    )
    return train_df, val_df, test_df


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train(config, fusion_type):
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, _ = prepare_splits(config)

    train_dataset = MemeSentimentDataset(train_df, config, split="train")
    val_dataset = MemeSentimentDataset(val_df, config, split="val")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = MultimodalSentimentModel(
        text_model_name=config["text_model"]["name"],
        image_model_name=config["image_model"]["name"],
        num_classes=len(config["labels"]["classes"]),
        fusion_type=fusion_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    best_f1 = 0.0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % config["training"]["log_every_n_steps"] == 0:
                progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(config["training"]["checkpoint_dir"], f"best_{fusion_type}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path}")

    print(f"Training complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="text_only",
        choices=["text_only", "image_only", "concat", "attention"],
        help="Which model variant to train — start with text_only as your baseline.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.fusion_type)
