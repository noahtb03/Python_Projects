"""
data_cleaning.py
----------------
Loads raw data, merges, cleans, deduplicates, and returns
clean DataFrames ready for training.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import (
    CONVERSATION_FILE, LABEL_FILE,
    CONVERSATION_ID_COL, LABEL_ID_COL, TEXT_COL,
    INCLUDE_COL, INCLUDE_YES, INCLUDE_NO,
    BINARY_FIELDS, CATEGORICAL_FIELDS, ALL_FIELDS,
    CONDITIONAL_FIELDS, FORCE_NO_STRINGS,
)


def normalize_text(s):
    """
    Clean raw text by collapsing all whitespace into single spaces.
    Handles None, numeric, and messy inputs so the tokenizer gets
    consistent, clean text every time.
    """
    if s is None:
        return ""
    s = str(s)
    return " ".join(s.split()).strip()


def clean_label(val):
    """
    Clean a single label value from the spreadsheet.
    Returns None for empty cells or blank strings so they can
    be converted to IGNORE_LABEL (-1) later, telling the model
    to skip this label during training.
    """
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val).strip()


def load_and_merge(conv_path=None, label_path=None):
    """
    Load the conversation file and label file from Excel, normalize
    their ID columns so they match, and inner-join them together.
    Only conversations that have labels (and vice versa) are kept.
    """
    conv_path = conv_path or CONVERSATION_FILE
    label_path = label_path or LABEL_FILE

    print(f"Loading conversations from {conv_path}...")
    conv_df = pd.read_excel(conv_path)

    print(f"Loading labels from {label_path}...")
    label_df = pd.read_excel(label_path)

    # Normalize IDs to string and strip whitespace so they match during merge
    conv_df[CONVERSATION_ID_COL] = conv_df[CONVERSATION_ID_COL].astype(str).str.strip()
    label_df[LABEL_ID_COL] = label_df[LABEL_ID_COL].astype(str).str.strip()

    # Inner join: only keep rows where the ID exists in both files
    df = conv_df.merge(
        label_df,
        left_on=CONVERSATION_ID_COL,
        right_on=LABEL_ID_COL,
        how="inner",
    )
    print(f"Merged dataset: {len(df)} rows")
    return df


def clean_include_labels(df):
    """
    Convert the Include column from raw strings ("Yes", "yes", "YES")
    into clean binary integers (1 or 0). Also applies any force-no
    rules — text patterns that should always be labeled as not included,
    regardless of what the original label says.
    """
    df = df.copy()
    # Chain: convert to string → strip whitespace → lowercase → map to 1/0
    df[INCLUDE_COL] = (
        df[INCLUDE_COL]
        .astype(str).str.strip().str.lower()
        .map({INCLUDE_YES: 1, INCLUDE_NO: 0})
        .fillna(0).astype(int)
    )
    # Override: any conversation containing a force-no pattern gets set to 0
    for pattern in FORCE_NO_STRINGS:
        mask = df[TEXT_COL].astype(str).str.contains(pattern, regex=False)
        df.loc[mask, INCLUDE_COL] = 0
    return df


def clean_text(df):
    """
    Apply normalize_text to every row, creating a clean TextNorm column.
    Remove any rows where the text is empty after cleaning — there's
    nothing for the model to learn from.
    """
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df["TextNorm"] = df[TEXT_COL].apply(normalize_text)
    df = df[df["TextNorm"].str.len() > 0].reset_index(drop=True)
    return df


def deduplicate(df):
    """
    Remove rows with duplicate conversation IDs, keeping the first
    occurrence. This prevents the same conversation from appearing
    with conflicting labels, which would guarantee errors no matter
    what the model predicts.
    """
    before = len(df)
    df = df.drop_duplicates(subset=CONVERSATION_ID_COL).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate records")
    return df


def clean_classification_labels(df):
    """
    Normalize all 9 label columns for the multi-task classifier.
    Binary fields get mapped to consistent "Yes"/"No".
    Categorical fields just get whitespace cleaned.
    Conditional logic is applied — e.g., if ConcernRaised is No,
    ConcernDetails gets blanked out since there's nothing to predict.
    """
    df = df.copy()
    # Normalize binary fields to consistent Yes/No
    for col in BINARY_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(clean_label)
            df[col] = df[col].str.lower().map({"yes": "Yes", "no": "No"})

    # Clean categorical fields (no mapping — categories discovered later)
    for col in CATEGORICAL_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(clean_label)

    # Apply conditional logic from config
    for dependent, (controller, skip_value) in CONDITIONAL_FIELDS.items():
        if controller in df.columns and dependent in df.columns:
            mask = df[controller] == skip_value
            df.loc[mask, dependent] = ""
            df[dependent] = df[dependent].fillna("")

    return df


def prepare_include_data():
    """
    Full data preparation pipeline for the include model.
    Loads both files, merges, cleans labels and text, removes
    duplicates. Returns a DataFrame with all conversations and
    their binary include labels ready for training.
    """
    df = load_and_merge()
    df = clean_include_labels(df)
    df = clean_text(df)
    df = deduplicate(df)
    print(f"\nInclude data ready: {len(df)} rows")
    print(f"Class distribution:\n{df[INCLUDE_COL].value_counts()}")
    return df


def prepare_classifier_data():
    """
    Full data preparation pipeline for the multi-task classifier.
    Same as prepare_include_data but with two extra steps: filters
    to only Include=Yes conversations and cleans the 9 label columns.
    The classifier only trains on included records.
    """
    df = load_and_merge()
    df = clean_include_labels(df)
    df = clean_text(df)
    df = deduplicate(df)
    # Keep only included records — classifier never sees excluded ones
    df = df[df[INCLUDE_COL] == 1].reset_index(drop=True)
    df = clean_classification_labels(df)
    print(f"\nClassifier data ready: {len(df)} rows")
    for col in ALL_FIELDS:
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")
    return df