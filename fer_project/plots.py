"""All charts. Each function saves a PNG file and returns its path."""
import matplotlib

matplotlib.use("Agg")  # draw to files; works on Colab, servers and laptops alike
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def class_counts(counts, class_names, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(class_names, counts, color="#4C72B0")
    ax.set_title("Training images per class")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def sample_grid(x, labels, class_names, path, per_class=8):
    k = len(class_names)
    fig, axes = plt.subplots(k, per_class, figsize=(per_class, k * 1.1))
    for ci, c in enumerate(class_names):
        idx = np.where(labels == ci)[0][:per_class]
        for j in range(per_class):
            ax = axes[ci, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(x[idx[j], 0], cmap="gray")
        axes[ci, 0].set_title(c, fontsize=8, loc="left")
    fig.tight_layout()
    return _save(fig, path)


def image_row(images, title, path):
    fig, axes = plt.subplots(2, 8, figsize=(10, 3.0))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    return _save(fig, path)


def learning_curves(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(hist.epoch, hist.train_loss, label="train")
    axes[0].plot(hist.epoch, hist.val_loss, label="validation")
    axes[0].set_title("Loss (rising validation loss = overfitting)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(hist.epoch, hist.accuracy, label="accuracy")
    axes[1].plot(hist.epoch, hist.macro_f1, label="macro-F1")
    axes[1].set_title("Validation scores")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, path)


def confusion(cm, class_names, path):
    k = len(class_names)
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title("Confusion matrix (each row sums to 1)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return _save(fig, path)


def mistakes(x, true, pred, class_names, path):
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, t, p in zip(axes.flat, x, true, pred):
        ax.imshow(img[0], cmap="gray")
        ax.set_title(f"true: {class_names[t]}\npred: {class_names[p]}", fontsize=8)
    fig.suptitle("Most confident mistakes")
    fig.tight_layout()
    return _save(fig, path)


def photo_prediction(img_rgb, box, face, probs, class_names, path):
    x, y, w, h = box
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.3, 1, 1.6]})
    axes[0].imshow(img_rgb)
    axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, color="lime", lw=2))
    axes[0].axis("off")
    axes[1].imshow(face, cmap="gray")
    axes[1].set_title("what the model sees")
    axes[1].axis("off")
    order = np.argsort(probs)
    axes[2].barh([class_names[i] for i in order], probs[order], color="#4C72B0")
    axes[2].set_xlim(0, 1)
    axes[2].set_title(f"prediction: {class_names[probs.argmax()]} ({probs.max():.0%})")
    fig.tight_layout()
    return _save(fig, path)
