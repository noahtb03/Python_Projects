"""
models.py
---------
Model architecture definitions used by both pipelines.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from transformers import BertModel
from config import DROPOUT


class MultiTaskModel(nn.Module):
    """
    Shared BERT encoder with multiple classification heads.
    One BERT model reads the text and produces a single summary vector.
    That vector feeds into separate classification heads — one per field.
    Each head independently predicts its own field's label.
    """

    def __init__(self, model_name, task_num_labels):
        """
        Build the model architecture.
        Loads pretrained BERT, adds dropout, and creates one Linear
        classification head per field. Each head maps BERT's 768-dim
        output to the number of classes for that field.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(DROPOUT)

        hidden_size = self.bert.config.hidden_size
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, n_labels)
            for name, n_labels in task_num_labels.items()
        })
        self.task_names = list(task_num_labels.keys())

    def forward(self, input_ids, attention_mask, labels=None, ignore_label=-1):
        """
        Run one forward pass through the model.
        Takes tokenized text, runs it through BERT, then feeds the
        pooled output into every classification head. If labels are
        provided (during training), computes the average loss across
        all fields, skipping any labels set to ignore_label (-1).
        Returns the predictions (logits) and the loss.
        """
        # Run text through BERT's 12 transformer layers
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get the single vector summarizing the entire input
        pooled = self.dropout(outputs.pooler_output)

        # Feed the same vector into every classification head
        logits = {name: head(pooled) for name, head in self.heads.items()}

        # Compute loss only during training (when labels are provided)
        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, device=input_ids.device)
            active_tasks = 0
            for name in self.task_names:
                if name in labels:
                    task_labels = labels[name]
                    # Skip missing labels (marked as -1)
                    mask = task_labels != ignore_label
                    if mask.any():
                        task_loss = nn.CrossEntropyLoss()(
                            logits[name][mask], task_labels[mask]
                        )
                        loss = loss + task_loss
                        active_tasks += 1
            # Average across tasks so no single field dominates
            if active_tasks > 0:
                loss = loss / active_tasks

        return logits, loss