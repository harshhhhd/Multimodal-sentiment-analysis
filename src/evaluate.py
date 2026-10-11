"""
Evaluation script — loads a trained checkpoint and reports Accuracy,
Precision, Recall, macro-F1, and a confusion matrix on the test split.

Usage:
    python -m src.evaluate --fusion_type text_only --checkpoint checkpoints/best_text_only.pt
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

from src.dataset import MemeSentimentDataset
from src.models.fusion_model import MultimodalSentimentModel
from src.train import load_config, prepare_splits


def run_evaluation(config, fusion_type, checkpoint_path):
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    _, _, test_df = prepare_splits(config)
    test_dataset = MemeSentimentDataset(test_df, config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = MultimodalSentimentModel(
        text_model_name=config["text_model"]["name"],
        image_model_name=config["image_model"]["name"],
        num_classes=len(config["labels"]["classes"]),
        fusion_type=fusion_type,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    class_names = config["labels"]["classes"]
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 (macro):{f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_type", type=str, required=True,
                         choices=["text_only", "image_only", "concat", "attention"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg, args.fusion_type, args.checkpoint)
