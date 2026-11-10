"""
Training pipeline for ABSA fine-tuning.
Uses SemEval-2014 Task 4 Laptop/Restaurant dataset format.
Run with: python scripts/train_absa.py

Requires: pip install datasets transformers torch mlflow scikit-learn
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── Sample built-in training data ──────────────────────────────────────────
BUILTIN_SAMPLES = [
    # (review, aspect, label)
    ("The battery lasts all day.", "battery", "positive"),
    ("Battery drains within 3 hours.", "battery", "negative"),
    ("Battery life is acceptable.", "battery", "neutral"),
    ("The camera takes stunning photos.", "camera", "positive"),
    ("Camera quality is terrible in low light.", "camera", "negative"),
    ("The screen is bright and crisp.", "screen", "positive"),
    ("Display is washed out outdoors.", "screen", "negative"),
    ("Performance is fast and smooth.", "performance", "positive"),
    ("The phone lags constantly.", "performance", "negative"),
    ("Great value for the price.", "price", "positive"),
    ("Too expensive for what you get.", "price", "negative"),
    ("Build quality feels premium.", "build", "positive"),
    ("Plastic body feels cheap.", "build", "negative"),
    ("Software is smooth and feature-rich.", "software", "positive"),
    ("Too many bloatware apps.", "software", "negative"),
    ("Arrived on time, well packaged.", "delivery", "positive"),
    ("Delivery took 3 weeks.", "delivery", "negative"),
    ("Battery is okay, nothing special.", "battery", "neutral"),
    ("The price is fair but not exceptional.", "price", "neutral"),
    ("Screen brightness is average.", "screen", "neutral"),
]


class ABSADataset(Dataset):
    """Dataset for aspect-conditioned sentiment classification."""

    def __init__(self, samples: List[Tuple[str, str, str]], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        review, aspect, label = self.samples[idx]
        # Format: "review text [SEP] aspect"
        encoding = self.tokenizer(
            review,
            aspect,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(LABEL_MAP[label], dtype=torch.long),
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Setup MLflow if requested
    mlflow_run = None
    if args.use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("absa_fine_tuning")
            mlflow_run = mlflow.start_run()
            mlflow.log_params(vars(args))
            logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except ImportError:
            logger.warning("MLflow not installed — skipping experiment tracking.")

    # Load tokenizer and model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Prepare data
    all_samples = BUILTIN_SAMPLES.copy()

    # Optionally load external JSONL dataset
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading external dataset: {args.data_path}")
        with open(args.data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                # Expected format: {"review": str, "aspect": str, "sentiment": str}
                sentiment = item.get("sentiment", "neutral").lower()
                if sentiment not in LABEL_MAP:
                    continue
                all_samples.append((item["review"], item["aspect"], sentiment))

    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split]
    val_samples = all_samples[split:] or all_samples[:5]  # ensure val is non-empty

    logger.info(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = ABSADataset(train_samples, tokenizer, args.max_length)
    val_ds = ABSADataset(val_samples, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += outputs.loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                correct += (preds == labels).sum()
                total += len(labels)

        val_acc = correct / total if total > 0 else 0.0

        # Compute F1
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=list(LABEL_MAP.keys()), zero_division=0)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f}"
        )
        logger.info(f"\n{report}")

        if mlflow_run:
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_accuracy": val_acc,
                "macro_f1": macro_f1,
            }, step=epoch)

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(output_dir / "best_model"))
            tokenizer.save_pretrained(str(output_dir / "best_model"))
            logger.info(f"✔ Saved best model (val_acc={val_acc:.4f})")

    # Save final
    model.save_pretrained(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    summary = {
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "base_model": args.base_model,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training complete. Summary: {summary}")

    if mlflow_run:
        mlflow.log_artifact(str(output_dir / "training_summary.json"))
        mlflow.end_run()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an ABSA model")
    parser.add_argument("--base_model", default="distilbert-base-uncased")
    parser.add_argument("--data_path", default=None, help="Path to external JSONL dataset")
    parser.add_argument("--output_dir", default="models/fine_tuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("--mlflow_uri", default="data/mlruns")
    args = parser.parse_args()
    train(args)
