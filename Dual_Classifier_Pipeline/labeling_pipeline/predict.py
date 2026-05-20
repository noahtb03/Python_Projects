"""
predict.py
----------
Prediction functions for both models.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from config import (
    DEVICE, INCLUDE_MODEL_DIR, CLASSIFIER_MODEL_DIR, MAX_LEN, BATCH_SIZE,
    INCLUDE_THRESHOLD, ALL_FIELDS, CONDITIONAL_FIELDS, FORCE_NO_STRINGS,
    TEXT_COL,
)
from shared.models import MultiTaskModel
from shared.datasets import PredictionDataset
from shared.data_cleaning import normalize_text


def predict_include(df):
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df["TextNorm"] = df[TEXT_COL].apply(normalize_text)
    df = df[df["TextNorm"].str.len() > 0].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(INCLUDE_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(INCLUDE_MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    dataset = PredictionDataset(df["TextNorm"].tolist(), tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running include model"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    df["IncludeProb"] = all_probs
    df["IncludePredicted"] = (np.array(all_probs) >= INCLUDE_THRESHOLD).astype(int)

    for pattern in FORCE_NO_STRINGS:
        mask = df[TEXT_COL].astype(str).str.contains(pattern, regex=False)
        df.loc[mask, "IncludePredicted"] = 0

    included = df[df["IncludePredicted"] == 1].reset_index(drop=True)

    print(f"\nInclude model: {len(df)} total → {len(included)} included (threshold={INCLUDE_THRESHOLD})")
    return included


def predict_labels(df):
    meta_path = os.path.join(CLASSIFIER_MODEL_DIR, "classifier_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_num_labels = meta["task_num_labels"]
    label_maps = meta["label_maps"]
    inverse_maps = {field: {int(v): k for k, v in mapping.items()} for field, mapping in label_maps.items()}

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_DIR, use_fast=True)
    model = MultiTaskModel(CLASSIFIER_MODEL_DIR, task_num_labels)
    model.heads.load_state_dict(torch.load(os.path.join(CLASSIFIER_MODEL_DIR, "heads.pt"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = PredictionDataset(df["TextNorm"].tolist(), tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    raw_preds = {f: [] for f in ALL_FIELDS}
    raw_probs = {f: [] for f in ALL_FIELDS}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running classifier"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask)
            for field in ALL_FIELDS:
                probs = torch.softmax(logits[field], dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                raw_preds[field].extend(preds.tolist())
                raw_probs[field].extend(probs[np.arange(len(preds)), preds].tolist())

    predictions = {}
    for field in ALL_FIELDS:
        inv_map = inverse_maps[field]
        predictions[field] = [inv_map.get(p, "Unknown") for p in raw_preds[field]]

    for dependent, (controller, skip_value) in CONDITIONAL_FIELDS.items():
        if controller in predictions and dependent in predictions:
            for i in range(len(predictions[controller])):
                if predictions[controller][i] == skip_value:
                    predictions[dependent][i] = ""

    confidences = {f: [round(p, 4) for p in raw_probs[f]] for f in ALL_FIELDS}
    print(f"Classifier: {len(df)} records labeled")
    return predictions, confidences