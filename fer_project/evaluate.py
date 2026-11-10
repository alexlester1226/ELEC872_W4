"""Evaluate a trained model once on the held-out test set.

    python evaluate.py --checkpoint runs/<run name>/best.pt

Writes test_results.json, plots/confusion_matrix.png and plots/mistakes.png next to the checkpoint.

Typical results (a rough guide; they vary with settings and luck):
  original FER2013 labels: about 68–72% accuracy with ResNet-18
  FER+ labels:             about 83–87% accuracy
Human agreement on the original FER2013 labels is estimated at only about 65%, so those labels are noisy.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import plots
from augment import to_device
from data import load_data
from models import build_model
from utils import get_device, predict, scores


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the FER2013 test set")
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("--data-dir", default=None, help="override the data folder saved in the checkpoint")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="also predict on the mirrored face and average")
    args = parser.parse_args(argv)

    device = get_device()
    ckpt_path = Path(args.checkpoint)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["config"])
    cfg.data_dir = Path(args.data_dir or cfg.data_dir)
    print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, model {cfg.model}, labels {cfg.labels})")

    data, class_names = load_data(cfg)
    if class_names != ck["class_names"]:
        raise ValueError(f"Checkpoint classes {ck['class_names']} don't match the data's classes {class_names}")
    model = build_model(cfg.model, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    x_test, y_test = to_device({"test": data["test"]}, device)["test"]
    probs, labels, _ = predict(model, x_test, y_test, cfg.img_size, tta=args.tta)
    pred = probs.argmax(1)
    result = scores(probs, labels)

    print(f"\nTEST accuracy: {result['accuracy']:.4f} | TEST macro-F1: {result['macro_f1']:.4f}\n")
    print(classification_report(labels, pred, labels=range(len(class_names)), target_names=class_names,
                                digits=3, zero_division=0))

    out = ckpt_path.parent
    cm = confusion_matrix(labels, pred, labels=range(len(class_names)))
    plots.confusion(cm, class_names, out / "plots" / "confusion_matrix.png")

    wrong = np.where(pred != labels)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:16]     # most confident mistakes first
    if len(wrong):
        plots.mistakes(data["test"][0][wrong], labels[wrong], pred[wrong], class_names, out / "plots" / "mistakes.png")

    with open(out / "test_results.json", "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "epoch": ck["epoch"], "model": cfg.model, "labels": cfg.labels,
                   "test_images": int(len(labels)), **result,
                   "confusion_matrix": cm.tolist(), "class_names": class_names}, f, indent=2)
    print("Saved results and plots in", out)
    return result


if __name__ == "__main__":
    main()
