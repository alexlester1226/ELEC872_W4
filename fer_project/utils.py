"""Shared helpers: seeding, the weighted loss, prediction and scores."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from augment import batches, prepare


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no GPU found: training will be slow)")
    return device


def class_weights(counts, scheme, device):
    """Weighted loss, the AffectNet paper's fix for class imbalance.

    Rare classes ("disgust", "contempt") get larger weights, so mistakes on them cost more.
      inverse:      weight ∝ 1 / class size (as in the paper)
      sqrt_inverse: weight ∝ 1 / sqrt(class size), a gentler compromise
    Weights are scaled so the average weight per training image is 1.
    """
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if scheme == "none":
        w = np.ones_like(counts)
    elif scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(scheme)
    w = w / (w * counts).sum() * counts.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


def loss_fn(logits, targets, weight, label_smoothing):
    # targets are probability vectors (one-hot or FER+ vote shares); cross_entropy accepts both
    return F.cross_entropy(logits.float(), targets, weight=weight, label_smoothing=label_smoothing)


@torch.no_grad()
def predict(model, x, y, img_size, tta=True, batch_size=512, weight=None, label_smoothing=0.0):
    """Return (probabilities, true labels, mean loss) for a whole split held on the GPU."""
    model.eval()
    device_type = x.device.type
    probs, labels, total_loss = [], [], 0.0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            xin = prepare(xb, img_size, train=False)
            logits = model(xin).float()
            if tta:
                logits = (logits + model(xin.flip(3)).float()) / 2
        total_loss += loss_fn(logits, yb, weight, label_smoothing).item() * len(xb)
        probs.append(logits.softmax(1).cpu())
        labels.append(yb.argmax(1).cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, total_loss / len(labels)


def scores(probs, labels):
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0))}
