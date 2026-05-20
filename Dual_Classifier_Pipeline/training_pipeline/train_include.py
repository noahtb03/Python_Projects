"""
train_include.py
----------------
Trains the binary include/exclude classifier.
Handles class imbalance with WeightedRandomSampler.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, BertForSequenceClassification,
    DataCollatorWithPadding, get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from config import (
    DEVICE, BASE_MODEL_NAME, INCLUDE_MODEL_DIR, MAX_LEN, BATCH_SIZE,
    INCLUDE_EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO, GRAD_CLIP,
    NUM_WORKERS, PIN_MEMORY, USE_MIXED_PRECISION, INCLUDE_COL,
)
from shared.datasets import IncludeDataset


def find_best_threshold(y_true, probs):
    """
    Search for the threshold that maximizes F1 score.
    Tests 19 evenly-spaced values from 0.05 to 0.95 and keeps
    whichever one gives the best balance of precision and recall.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


@torch.no_grad()  # Disables gradient tracking — saves memory, speeds up evaluation
def evaluate(model, loader, device):
    """
    Run the model on the validation set and collect predictions.
    Returns the true labels and the model's predicted probabilities
    so we can test different thresholds against them.
    """
    model.eval()  # Switches to eval mode — disables dropout
    all_probs, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # softmax converts raw scores to probabilities; [:, 1] takes the Yes probability
        probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train_include_model(df):
    """
    Train the binary include/exclude classifier.
    Splits data 80/20, handles class imbalance with weighted sampling,
    trains BERT with mixed precision, evaluates after each epoch,
    and saves the best model based on F1 score.
    """
    os.makedirs(INCLUDE_MODEL_DIR, exist_ok=True)

    texts = df["TextNorm"].tolist()
    labels = df[INCLUDE_COL].tolist()

    # Split into 80% training, 20% validation; stratify keeps class ratio in both splits
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Load pretrained BERT tokenizer and model with a fresh 2-class classification head
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=2
    ).to(DEVICE)

    # --- Handle class imbalance with weighted sampling ---
    # Minority class (Yes) gets higher weight so the model sees equal amounts of each class
    train_labels_np = np.array(train_labels, dtype=int)
    class_counts = np.bincount(train_labels_np, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels_np]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True,
    )

    # --- Create data loaders that feed batches to the model ---
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = IncludeDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = IncludeDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
    )

    # --- Optimizer, learning rate scheduler, and mixed precision scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * INCLUDE_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda", enabled=(USE_MIXED_PRECISION and DEVICE.type == "cuda"))

    best_val_f1, best_threshold = -1.0, 0.5

    # === TRAINING LOOP ===
    for epoch in range(INCLUDE_EPOCHS):
        model.train()  # Enable dropout and training behavior
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # Clear leftover gradients

        pbar = tqdm(train_loader, desc=f"Include — Epoch {epoch+1}/{INCLUDE_EPOCHS}")
        for step, batch in enumerate(pbar, start=1):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels_batch = batch["labels"].to(DEVICE, non_blocking=True)

            # Forward pass with mixed precision (float16 where safe)
            with autocast("cuda", enabled=(USE_MIXED_PRECISION and DEVICE.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = out.loss

            # Backward pass: compute gradients
            scaler.scale(loss).backward()
            total_loss += loss.item()

            # Clip gradients to prevent exploding updates, then step optimizer
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Adjust learning rate
            optimizer.zero_grad(set_to_none=True)  # Clear gradients for next batch

            pbar.set_postfix(loss=(total_loss / step))

        # === EVALUATE AFTER EACH EPOCH ===
        avg_loss = total_loss / max(len(train_loader), 1)
        y_true, probs = evaluate(model, val_loader, DEVICE)
        threshold, val_f1 = find_best_threshold(y_true, probs)
        preds = (probs >= threshold).astype(int)

        val_acc = accuracy_score(y_true, preds)
        val_prec = precision_score(y_true, preds, zero_division=0)
        val_rec = recall_score(y_true, preds, zero_division=0)

        print(
            f"\n  Epoch {epoch+1} | loss={avg_loss:.4f}"
            f" | acc={val_acc:.4f} f1={val_f1:.4f}"
            f" prec={val_prec:.4f} rec={val_rec:.4f} | thr={threshold:.2f}\n"
        )

        # Save the model if this epoch produced the best F1 so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            model.save_pretrained(INCLUDE_MODEL_DIR)
            tokenizer.save_pretrained(INCLUDE_MODEL_DIR)
            # Save metadata so the prediction script knows the threshold and config
            meta = {
                "best_val_f1": float(best_val_f1),
                "best_threshold": float(best_threshold),
                "max_len": MAX_LEN,
                "model_name": BASE_MODEL_NAME,
                "class_counts_train": class_counts.tolist(),
            }
            with open(os.path.join(INCLUDE_MODEL_DIR, "include_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"✅ Include model saved | F1: {best_val_f1:.4f} | Threshold: {best_threshold:.2f}")
    return best_threshold, best_val_f1