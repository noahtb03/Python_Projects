"""
train_classifier.py
-------------------
Trains the multi-task classifier with separate heads per field.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from config import (
    DEVICE, BASE_MODEL_NAME, CLASSIFIER_MODEL_DIR, MAX_LEN, BATCH_SIZE,
    CLASSIFIER_EPOCHS, LR, WEIGHT_DECAY, WARMUP_RATIO, GRAD_CLIP,
    NUM_WORKERS, PIN_MEMORY, USE_MIXED_PRECISION,
    BINARY_FIELDS, CATEGORICAL_FIELDS, ALL_FIELDS,
    CONDITIONAL_FIELDS, IGNORE_LABEL,
)
from shared.models import MultiTaskModel
from shared.datasets import MultiTaskDataset, MultiTaskCollator


def build_label_maps(df):
    """
    Create mappings from string labels to integer indices.
    Binary fields always map Yes=0, No=1.
    Categorical fields discover all unique values from the data,
    sort them alphabetically, and assign each an index.
    These maps are saved to JSON so predictions can be decoded later.
    """
    label_maps = {}
    for col in BINARY_FIELDS:
        label_maps[col] = {"Yes": 0, "No": 1}
    for col in CATEGORICAL_FIELDS:
        if col in df.columns:
            # Find all unique non-empty values and assign each an index
            unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
            unique_vals = [v for v in unique_vals if v != ""]
            label_maps[col] = {v: i for i, v in enumerate(unique_vals)}
        else:
            label_maps[col] = {}
    return label_maps


def encode_labels(df, label_maps):
    """
    Convert every label in the DataFrame from strings to integers
    using the label maps. Any value not in the map (including blanks)
    becomes IGNORE_LABEL (-1), which tells the loss function to skip
    that label during training rather than learning from bad data.
    """
    encoded = {}
    for col in ALL_FIELDS:
        if col in label_maps and col in df.columns:
            encoded[col] = df[col].map(label_maps[col]).fillna(IGNORE_LABEL).astype(int).tolist()
        else:
            encoded[col] = [IGNORE_LABEL] * len(df)
    return encoded


@torch.no_grad()  # No gradient tracking needed during evaluation
def evaluate(model, loader, device):
    """
    Run the multi-task model on the validation set.
    Collects predictions for all 9 fields, skipping any with
    IGNORE_LABEL. Returns per-field accuracy and weighted F1.
    """
    model.eval()  # Disable dropout for consistent evaluation
    all_preds = {f: [] for f in ALL_FIELDS}
    all_labels = {f: [] for f in ALL_FIELDS}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Build label dict for all 9 fields
        labels_batch = {}
        for f in ALL_FIELDS:
            key = f"label_{f}"
            if key in batch:
                labels_batch[f] = batch[key].to(device)
        logits, _ = model(input_ids, attention_mask, labels=labels_batch)
        for f in ALL_FIELDS:
            key = f"label_{f}"
            if key in batch:
                task_labels = batch[key].numpy()
                # argmax picks the class with the highest score
                task_preds = torch.argmax(logits[f], dim=-1).cpu().numpy()
                # Only keep predictions where the label isn't -1 (missing)
                mask = task_labels != IGNORE_LABEL
                all_labels[f].extend(task_labels[mask].tolist())
                all_preds[f].extend(task_preds[mask].tolist())

    results = {}
    for f in ALL_FIELDS:
        if len(all_labels[f]) > 0:
            acc = accuracy_score(all_labels[f], all_preds[f])
            f1 = f1_score(all_labels[f], all_preds[f], average="weighted", zero_division=0)
            results[f] = {"accuracy": acc, "f1_weighted": f1}
        else:
            results[f] = {"accuracy": 0.0, "f1_weighted": 0.0}
    return results


def train_classifier(df):
    """
    Train the multi-task classifier with 9 prediction heads.
    Similar to train_include but uses our custom MultiTaskModel,
    handles 9 label fields simultaneously, and saves the best model
    based on average F1 across all fields.
    """
    os.makedirs(CLASSIFIER_MODEL_DIR, exist_ok=True)

    # Discover all categories and create string-to-integer mappings
    label_maps = build_label_maps(df)
    print("\nLabel maps:")
    for field, mapping in label_maps.items():
        print(f"  {field}: {len(mapping)} classes")

    # Convert all string labels to integers; missing values become -1
    encoded_labels = encode_labels(df, label_maps)
    # Build dict of how many classes each field has (needed to create the heads)
    task_num_labels = {f: len(label_maps[f]) for f in ALL_FIELDS if len(label_maps[f]) > 0}

    # Split by index so we can split all 9 label lists consistently
    texts = df["TextNorm"].tolist()
    indices = list(range(len(texts)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = {f: [encoded_labels[f][i] for i in train_idx] for f in ALL_FIELDS}
    val_labels = {f: [encoded_labels[f][i] for i in val_idx] for f in ALL_FIELDS}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    # Use our custom multi-head model instead of BertForSequenceClassification
    model = MultiTaskModel(BASE_MODEL_NAME, task_num_labels).to(DEVICE)
    # Custom collator that handles 9 label fields
    collator = MultiTaskCollator(tokenizer, ALL_FIELDS)

    train_dataset = MultiTaskDataset(train_texts, train_labels, tokenizer, MAX_LEN, ALL_FIELDS)
    val_dataset = MultiTaskDataset(val_texts, val_labels, tokenizer, MAX_LEN, ALL_FIELDS)

    # No weighted sampler here — classifier data is already filtered to Include=Yes
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and DEVICE.type == "cuda"),
    )

    # Same optimizer, scheduler, and scaler setup as include model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * CLASSIFIER_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda", enabled=(USE_MIXED_PRECISION and DEVICE.type == "cuda"))

    best_avg_f1 = -1.0

    # === TRAINING LOOP ===
    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Classifier — Epoch {epoch+1}/{CLASSIFIER_EPOCHS}")
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            # Build label dict for all 9 fields from the batch
            labels_batch = {}
            for f in ALL_FIELDS:
                key = f"label_{f}"
                if key in batch:
                    labels_batch[f] = batch[key].to(DEVICE, non_blocking=True)

            # Forward pass — model computes loss across all 9 tasks internally
            with autocast("cuda", enabled=(USE_MIXED_PRECISION and DEVICE.type == "cuda")):
                logits, loss = model(input_ids, attention_mask, labels=labels_batch)

            # Backward pass + optimizer step (same as include model)
            scaler.scale(loss).backward()
            total_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=(total_loss / step))

        # === EVALUATE AFTER EACH EPOCH ===
        avg_loss = total_loss / max(len(train_loader), 1)
        results = evaluate(model, val_loader, DEVICE)

        print(f"\n  Epoch {epoch+1} | loss={avg_loss:.4f}")
        f1_scores = []
        for f in ALL_FIELDS:
            r = results[f]
            print(f"    {f:30s} | acc={r['accuracy']:.4f}  f1={r['f1_weighted']:.4f}")
            f1_scores.append(r["f1_weighted"])

        # Use average F1 across all 9 fields to decide which model version to keep
        avg_f1 = np.mean(f1_scores)
        print(f"    {'AVERAGE':30s} | f1={avg_f1:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            # Save in 3 parts: BERT encoder, classification heads, and metadata
            model.bert.save_pretrained(CLASSIFIER_MODEL_DIR)
            tokenizer.save_pretrained(CLASSIFIER_MODEL_DIR)
            torch.save(model.heads.state_dict(), os.path.join(CLASSIFIER_MODEL_DIR, "heads.pt"))

            # Save label maps so the prediction script can decode numbers back to labels
            meta = {
                "best_avg_f1": float(best_avg_f1),
                "max_len": MAX_LEN,
                "model_name": BASE_MODEL_NAME,
                "task_num_labels": task_num_labels,
                "label_maps": label_maps,
                "binary_fields": BINARY_FIELDS,
                "categorical_fields": CATEGORICAL_FIELDS,
                "all_fields": ALL_FIELDS,
                "conditional_fields": CONDITIONAL_FIELDS,
                "epoch": epoch + 1,
            }
            with open(os.path.join(CLASSIFIER_MODEL_DIR, "classifier_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            with open(os.path.join(CLASSIFIER_MODEL_DIR, "val_results.json"), "w") as f:
                json.dump(results, f, indent=2)

            print(f"    ✅ Saved best model (avg F1={best_avg_f1:.4f})")

    print(f"\n✅ Classifier saved | Best avg F1: {best_avg_f1:.4f}")
    return best_avg_f1, label_maps