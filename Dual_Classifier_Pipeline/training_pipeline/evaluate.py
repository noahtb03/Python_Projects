"""
evaluate.py
-----------
Evaluates both models and generates results files.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import (
    DEVICE, INCLUDE_MODEL_DIR, CLASSIFIER_MODEL_DIR, RESULTS_DIR,
    MAX_LEN, BATCH_SIZE, INCLUDE_COL, EVAL_THRESHOLDS,
    BINARY_FIELDS, ALL_FIELDS, IGNORE_LABEL,
)
from shared.models import MultiTaskModel
from shared.datasets import PredictionDataset


def evaluate_include_model(df):
    """
    Load the saved include model and test it across multiple thresholds.
    For each threshold, computes accuracy, precision, recall, and F1
    to show the tradeoff between catching real cases and avoiding
    false alarms. Saves results to an Excel file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the saved model and run predictions on the full dataset
    tokenizer = AutoTokenizer.from_pretrained(INCLUDE_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(INCLUDE_MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    dataset = PredictionDataset(df["TextNorm"].tolist(), tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Collect the model's Yes probability for every conversation
    all_probs = []
    all_labels = df[INCLUDE_COL].tolist()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating include model"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Test each threshold and record the metrics
    results = []
    for thr in EVAL_THRESHOLDS:
        preds = (all_probs >= thr).astype(int)
        results.append({
            "Threshold": thr,
            "Accuracy": round(accuracy_score(all_labels, preds), 4),
            "Precision": round(precision_score(all_labels, preds, zero_division=0), 4),
            "Recall": round(recall_score(all_labels, preds, zero_division=0), 4),
            "F1": round(f1_score(all_labels, preds, zero_division=0), 4),
        })
        print(f"\n  Threshold {thr:.2f}")
        print(f"    Accuracy : {results[-1]['Accuracy']}")
        print(f"    Precision: {results[-1]['Precision']}")
        print(f"    Recall   : {results[-1]['Recall']}")
        print(f"    F1       : {results[-1]['F1']}")

    output_path = os.path.join(RESULTS_DIR, "include_threshold_comparison.xlsx")
    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"\n✅ Threshold results saved to {output_path}")


def evaluate_classifier(df, label_maps):
    """
    Load the saved multi-task classifier and compare its predictions
    against the actual human labels. Computes accuracy, precision,
    recall, and F1 for each of the 9 fields. Uses binary averaging
    for Yes/No fields and weighted averaging for categorical fields.
    Saves per-field metrics to an Excel file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load metadata and build inverse maps (index → label string)
    meta_path = os.path.join(CLASSIFIER_MODEL_DIR, "classifier_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_num_labels = meta["task_num_labels"]
    # Reverse the label maps: {0: "Yes", 1: "No"} instead of {"Yes": 0, "No": 1}
    inverse_maps = {field: {int(v): k for k, v in mapping.items()} for field, mapping in label_maps.items()}

    # Load model: BERT encoder + classification heads
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_DIR, use_fast=True)
    model = MultiTaskModel(CLASSIFIER_MODEL_DIR, task_num_labels)
    model.heads.load_state_dict(torch.load(os.path.join(CLASSIFIER_MODEL_DIR, "heads.pt"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = PredictionDataset(df["TextNorm"].tolist(), tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    raw_preds = {f: [] for f in ALL_FIELDS}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating classifier"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask)
            for field in ALL_FIELDS:
                preds = torch.argmax(logits[field], dim=-1).cpu().numpy()
                raw_preds[field].extend(preds.tolist())

    decoded_preds = {}
    for field in ALL_FIELDS:
        inv_map = inverse_maps[field]
        decoded_preds[field] = [inv_map.get(p, "Unknown") for p in raw_preds[field]]

    metrics_rows = []
    for field in ALL_FIELDS:
        if field not in df.columns:
            continue
        actual = df[field].tolist()
        predicted = decoded_preds[field]
        pairs = [(p, a) for p, a in zip(predicted, actual) if a is not None and str(a).strip() != ""]
        if not pairs:
            continue

        y_pred = [p for p, a in pairs]
        y_true = [a for p, a in pairs]
        acc = accuracy_score(y_true, y_pred)

        if field in BINARY_FIELDS:
            prec = precision_score(y_true, y_pred, average="binary", pos_label="Yes", zero_division=0)
            rec = recall_score(y_true, y_pred, average="binary", pos_label="Yes", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="binary", pos_label="Yes", zero_division=0)
        else:
            prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics_rows.append({"Field": field, "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4)})
        print(f"\n  {field}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1       : {f1:.4f}")

    output_path = os.path.join(RESULTS_DIR, "classifier_metrics.xlsx")
    pd.DataFrame(metrics_rows).to_excel(output_path, index=False)
    print(f"\n✅ Classifier metrics saved to {output_path}")