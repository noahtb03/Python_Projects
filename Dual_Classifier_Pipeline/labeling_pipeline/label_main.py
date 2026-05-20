"""
label_main.py
-------------
Labels new, unlabeled data using trained models:
    1. Run include model to filter records
    2. Run classifier to label included records
    3. Output clean labeled file
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import CONVERSATION_ID_COL, ALL_FIELDS
from labeling_pipeline.predict import predict_include, predict_labels


# =====================================================================
# INPUT / OUTPUT — update these for each new batch of data
# =====================================================================

INPUT_FILE  = "training_data/conversations.xlsx"
OUTPUT_FILE = "results/labeled_output.xlsx"


def main():
    print("=" * 60)
    print("LABELING PIPELINE")
    print("=" * 60)

    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    df[CONVERSATION_ID_COL] = df[CONVERSATION_ID_COL].astype(str).str.strip()
    print(f"Total records: {len(df)}")

    # ---- Stage 1: Include filter ----
    print("\n--- Stage 1: Include Model ---")
    included_df = predict_include(df)

    if len(included_df) == 0:
        print("\nNo records passed the include filter.")
        return

    # ---- Stage 2: Classifier ----
    print("\n--- Stage 2: Classifier ---")
    predictions, confidences = predict_labels(included_df)

    # ---- Build output ----
    output_df = pd.DataFrame({
        CONVERSATION_ID_COL: included_df[CONVERSATION_ID_COL].tolist()
    })

    if "Date" in included_df.columns:
        output_df["Date"] = included_df["Date"].tolist()
    if "Origin" in included_df.columns:
        output_df["Origin"] = included_df["Origin"].tolist()

    for field in ALL_FIELDS:
        output_df[field] = predictions[field]
    for field in ALL_FIELDS:
        output_df[f"{field}_Confidence"] = confidences[field]

    output_df["Include"] = "Yes"

    output_df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"LABELING COMPLETE")
    print(f"  Records labeled: {len(output_df)}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()