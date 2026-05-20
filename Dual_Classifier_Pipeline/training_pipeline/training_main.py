"""
training_main.py
----------------
Runs the full training pipeline:
    1. Clean and prepare data
    2. Train the include model
    3. Train the multi-task classifier
    4. Evaluate both models
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.data_cleaning import prepare_include_data, prepare_classifier_data
from training_pipeline.train_include import train_include_model
from training_pipeline.train_classifier import train_classifier
from training_pipeline.evaluate import evaluate_include_model, evaluate_classifier


def main():
    """
    Orchestrates the full training pipeline in three stages.
    Each stage calls functions from other modules — this file
    doesn't do any work itself, it just coordinates the order.
    """

    # ---- Stage 1: Clean all data and train the binary include model ----
    print("=" * 60)
    print("STAGE 1: Training Include Model")
    print("=" * 60)

    include_df = prepare_include_data()           # Load, merge, clean, deduplicate
    best_threshold, best_f1 = train_include_model(include_df)  # Train and save

    # ---- Stage 2: Filter to Include=Yes, train the 9-head classifier ----
    print("\n" + "=" * 60)
    print("STAGE 2: Training Multi-Task Classifier")
    print("=" * 60)

    classifier_df = prepare_classifier_data()     # Same data but filtered to included only
    best_avg_f1, label_maps = train_classifier(classifier_df)  # Train and save

    # ---- Stage 3: Load both saved models and measure performance ----
    print("\n" + "=" * 60)
    print("STAGE 3: Evaluating Both Models")
    print("=" * 60)

    print("\n--- Include Model Threshold Comparison ---")
    evaluate_include_model(include_df)             # Test multiple thresholds

    print("\n--- Classifier Metrics ---")
    evaluate_classifier(classifier_df, label_maps) # Per-field accuracy, precision, recall, F1

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Include threshold: {best_threshold:.2f}")
    print(f"  Include best F1:   {best_f1:.4f}")
    print(f"  Classifier avg F1: {best_avg_f1:.4f}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()