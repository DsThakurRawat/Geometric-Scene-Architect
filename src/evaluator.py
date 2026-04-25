"""
src/evaluator.py — Segmentation quality evaluation.

Computes precision, recall, F1 and a confusion matrix by comparing
predicted labels against ground-truth labels for synthetic data.

Usage (standalone):
    python -m src.evaluator --predicted outputs/segmentation_report.json \
                            --ground-truth data/synthetic/room_01_labels.json
"""
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


# All recognised labels in the pipeline
ALL_LABELS = [
    "floor", "ceiling", "wall",
    "furniture", "tall_furniture", "small_object",
    "high_fixture", "horizontal_surface",
    "unknown", "noise",
]


class SegmentationEvaluator:
    """
    Evaluates segmentation quality by comparing predicted vs. ground-truth labels.
    Works with both plane labels and object-cluster labels.
    """

    def __init__(self, labels: List[str] = None):
        self.labels = labels or ALL_LABELS

    def confusion_matrix(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Builds a confusion matrix as a nested dict:
            matrix[true_label][pred_label] = count
        """
        matrix: Dict[str, Dict[str, int]] = {
            t: {p: 0 for p in self.labels} for t in self.labels
        }

        for true, pred in zip(y_true, y_pred):
            if true in matrix and pred in matrix[true]:
                matrix[true][pred] += 1

        return matrix

    def per_class_metrics(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes precision, recall, F1 for each label.

        Returns:
            { label: {"precision": p, "recall": r, "f1": f, "support": n} }
        """
        matrix = self.confusion_matrix(y_true, y_pred)
        metrics: Dict[str, Dict[str, float]] = {}

        for label in self.labels:
            tp = matrix[label][label]
            # FP = predicted as `label` but true is something else
            fp = sum(matrix[t][label] for t in self.labels if t != label)
            # FN = true is `label` but predicted as something else
            fn = sum(matrix[label][p] for p in self.labels if p != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            support = tp + fn  # number of actual instances of this label

            metrics[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        return metrics

    def overall_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """Global accuracy: fraction of correct predictions."""
        if not y_true:
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return round(correct / len(y_true), 4)

    def evaluate(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict:
        """
        Full evaluation: accuracy + per-class metrics + confusion matrix.
        """
        return {
            "accuracy": self.overall_accuracy(y_true, y_pred),
            "per_class": self.per_class_metrics(y_true, y_pred),
            "confusion_matrix": self.confusion_matrix(y_true, y_pred),
        }

    def print_report(self, y_true: List[str], y_pred: List[str]) -> None:
        """Prints a human-readable classification report."""
        result = self.evaluate(y_true, y_pred)

        print(f"\n{'='*60}")
        print(f"  Segmentation Evaluation Report")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {result['accuracy']:.1%}")
        print(f"{'─'*60}")
        print(f"  {'Label':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        print(f"{'─'*60}")

        for label in self.labels:
            m = result["per_class"].get(label, {})
            if m.get("support", 0) > 0 or any(
                result["confusion_matrix"][t].get(label, 0) > 0
                for t in self.labels
            ):
                print(
                    f"  {label:<20} {m.get('precision', 0):.4f} "
                    f"{m.get('recall', 0):.4f} {m.get('f1', 0):.4f} "
                    f"{m.get('support', 0):>8}"
                )
        print(f"{'='*60}\n")


def evaluate_from_files(pred_path: str, gt_path: str) -> Dict:
    """
    Loads predicted and ground-truth JSON reports and evaluates.
    Expects both files to have 'structural_planes' and 'objects' lists
    with a 'label' field.
    """
    with open(pred_path) as f:
        pred = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    y_pred = [p["label"] for p in pred.get("structural_planes", [])]
    y_pred += [o["label"] for o in pred.get("objects", [])]

    y_true = [p["label"] for p in gt.get("structural_planes", [])]
    y_true += [o["label"] for o in gt.get("objects", [])]

    evaluator = SegmentationEvaluator()
    return evaluator.evaluate(y_true, y_pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation quality")
    parser.add_argument("--predicted", required=True, help="Path to predicted report JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground-truth JSON")
    args = parser.parse_args()

    with open(args.predicted) as f:
        pred = json.load(f)
    with open(args.ground_truth) as f:
        gt = json.load(f)

    y_pred = [p["label"] for p in pred.get("structural_planes", [])]
    y_pred += [o["label"] for o in pred.get("objects", [])]
    y_true = [p["label"] for p in gt.get("structural_planes", [])]
    y_true += [o["label"] for o in gt.get("objects", [])]

    evaluator = SegmentationEvaluator()
    evaluator.print_report(y_true, y_pred)
