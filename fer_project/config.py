"""All settings in one place. Every setting can be changed from the command line, e.g.

    python train.py --model alexnet --class-weighting inverse --epochs 30
"""
import argparse
from pathlib import Path

DEFAULTS = dict(
    # ---- data ----
    data_dir="data",            # where FER2013 is looked for / downloaded to
    labels="ferplus",           # "ferplus" (8 classes, cleaner labels) or "fer2013" (original 7 classes)
    ferplus_mode="majority",    # "majority" (one label per image) or "soft" (train on the annotators' vote shares)
    # ---- model ----
    model="resnet18",           # "resnet18", "alexnet" (as in the AffectNet paper) or "small_cnn" (from scratch)
    pretrained=True,            # start from ImageNet weights (ignored for small_cnn)
    img_size=128,               # FER2013 images are 48x48; pretrained networks work better on upscaled images
    # ---- training ----
    epochs=25,
    batch_size=128,
    lr=5e-4,
    weight_decay=1e-4,
    class_weighting="sqrt_inverse",  # "inverse" (paper-style), "sqrt_inverse" (gentler) or "none"
    label_smoothing=0.1,
    augment=True,               # random flip / rotation / shift / brightness / erasing
    tta=True,                   # test-time augmentation: also predict on the mirrored face and average
    select_metric="macro_f1",   # validation score used to pick the best checkpoint: "macro_f1" or "accuracy"
    seed=42,
    # ---- extras for Exercise 3 ----
    mixup=0.0,                  # mixup alpha (e.g. 0.2); 0 = off
    cutmix=0.0,                 # CutMix alpha (e.g. 1.0); 0 = off
    mix_prob=0.5,               # fraction of batches that get mixup/CutMix
    ema=0.0,                    # EMA decay of the weights (e.g. 0.999); 0 = off
    tag="",                     # free text added to the run folder name
    # ---- output ----
    out_dir="runs",             # checkpoints, plots and results go to out_dir/<run name>/
)

HELP = dict(
    data_dir="folder containing FER2013 (fer2013.csv, the Kaggle zip, or train/ and test/ folders)",
    labels="ferplus: 8 classes, cleaner labels | fer2013: the original 7 classes",
    ferplus_mode="majority: one label per image | soft: train on the annotators' vote shares",
    model="alexnet is the AffectNet paper's network; small_cnn trains from scratch",
    pretrained="start from ImageNet weights",
    img_size="size images are upscaled to (FER2013 images are 48x48)",
    epochs="number of passes over the training set",
    batch_size="images per training step",
    lr="peak learning rate",
    weight_decay="AdamW weight decay",
    class_weighting="weighted loss for class imbalance; inverse is the paper's choice",
    label_smoothing="label smoothing for the loss",
    augment="random flip / rotation / shift / brightness / erasing during training",
    tta="also predict on the mirrored face and average",
    select_metric="validation score used to pick the best checkpoint",
    seed="random seed",
    mixup="mixup alpha, e.g. 0.2 (0 = off)",
    cutmix="CutMix alpha, e.g. 1.0 (0 = off); with both on, each mixed batch picks one at random",
    mix_prob="probability that a batch is mixed",
    ema="decay of the exponential moving average of the weights, e.g. 0.999 (0 = off)",
    tag="extra text for the run folder name",
    out_dir="results go to out_dir/<run name>/",
)

CHOICES = dict(
    labels=["ferplus", "fer2013"],
    ferplus_mode=["majority", "soft"],
    model=["resnet18", "resnet18_small", "alexnet", "facenet", "small_cnn"],
    class_weighting=["inverse", "sqrt_inverse", "none"],
    select_metric=["macro_f1", "accuracy"],
)


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="AffectNet-style facial expression baseline on FER2013")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        help_text = f"{HELP.get(name, '')} (default: {default})"
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=HELP.get(name))
        else:
            parser.add_argument(flag, type=type(default), default=default, choices=CHOICES.get(name), help=help_text)
    cfg = parser.parse_args(argv)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.run_dir = Path(cfg.out_dir) / run_name(cfg)
    return cfg


def run_name(cfg):
    mode = f"_{cfg.ferplus_mode}" if cfg.labels == "ferplus" else ""
    name = f"{cfg.model}_{cfg.labels}{mode}_{cfg.img_size}px_{cfg.class_weighting}"
    # extra suffixes so different experiments never share (and resume from) the same folder
    if not cfg.pretrained and cfg.model != "small_cnn":
        name += "_scratch"
    if cfg.epochs != DEFAULTS["epochs"]:
        name += f"_e{cfg.epochs}"
    if cfg.mixup > 0:
        name += f"_mixup{cfg.mixup:g}"
    if cfg.cutmix > 0:
        name += f"_cutmix{cfg.cutmix:g}"
    if cfg.ema > 0:
        name += f"_ema{cfg.ema:g}"
    if cfg.tag:
        name += f"_{cfg.tag}"
    return name
