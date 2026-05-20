"""
config.py
---------
Central configuration for the entire pipeline.
All settings that change between projects live here.
"""

import torch

# =====================================================================
# DEVICE — automatically uses GPU if available
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# FILE PATHS — update these to match your project's data files
# =====================================================================

CONVERSATION_FILE = "training_data/conversations.xlsx"
LABEL_FILE        = "training_data/labels.xlsx"

INCLUDE_MODEL_DIR    = "saved_include_model"
CLASSIFIER_MODEL_DIR = "saved_classifier_model"
RESULTS_DIR          = "results"


# =====================================================================
# COLUMN NAMES — update these to match your data's column headers
# =====================================================================

# Unique identifier in the conversation file
CONVERSATION_ID_COL = "ConversationId"

# Unique identifier in the label file
LABEL_ID_COL = "ChatId"

# Column containing the text to classify
TEXT_COL = "Text"

# Column that determines include/exclude (stage 1)
INCLUDE_COL = "Include"

# What values in the include column mean Yes and No
INCLUDE_YES = "yes"
INCLUDE_NO  = "no"


# =====================================================================
# CLASSIFICATION FIELDS — update these to match your label columns
#
# BINARY_FIELDS:      columns with exactly two values (Yes/No)
# CATEGORICAL_FIELDS: columns with multiple possible values
#                     (categories auto-discovered from training data)
# =====================================================================

BINARY_FIELDS = [
    "ConcernRaised",
    "AgentFollowedProtocol",
    "EscalationNeeded",
    "ReferencedPriorContact",
    "ResolvedInSession",
    "SystemError",
]

CATEGORICAL_FIELDS = [
    "ContactReason",
    "ConcernDetails",
    "IssueCategory",
]

ALL_FIELDS = BINARY_FIELDS + CATEGORICAL_FIELDS


# =====================================================================
# CONDITIONAL FIELD LOGIC — optional
#
# If one field's value should be blank based on another field,
# define that here.
#
# Format: { "dependent_field": ("controlling_field", "skip_value") }
# Example: if ConcernRaised == "No", ConcernDetails should be blank
#
# Set to {} if not needed.
# =====================================================================

CONDITIONAL_FIELDS = {
    "ConcernDetails": ("ConcernRaised", "No"),
}


# =====================================================================
# FORCE RULES — optional
#
# Text patterns that always force the include prediction to No.
# Set to [] if not needed.
# =====================================================================

FORCE_NO_STRINGS = []


# =====================================================================
# MODEL HYPERPARAMETERS
# =====================================================================

MAX_LEN       = 512          # max tokens per input
BATCH_SIZE    = 8            # examples per batch (lower if out of memory)
LR            = 2e-5         # learning rate
WEIGHT_DECAY  = 0.01         # regularization
WARMUP_RATIO  = 0.06         # fraction of training for LR warmup
GRAD_CLIP     = 1.0          # max gradient norm
DROPOUT       = 0.1          # dropout on classification heads

# Epochs — passes through the training data
INCLUDE_EPOCHS    = 2
CLASSIFIER_EPOCHS = 2

# Performance
NUM_WORKERS         = 0      # dataloader workers (0 for compatibility)
PIN_MEMORY          = True
USE_MIXED_PRECISION = True

# Base pretrained model
BASE_MODEL_NAME = "bert-base-uncased"


# =====================================================================
# INCLUDE MODEL THRESHOLD
# =====================================================================

INCLUDE_THRESHOLD = 0.80

# Thresholds to test during evaluation
EVAL_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]


# =====================================================================
# INTERNAL — do not change
# =====================================================================

IGNORE_LABEL = -1