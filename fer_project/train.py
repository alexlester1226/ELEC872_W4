"""Train the AffectNet-style baseline on FER2013, then evaluate it on the test set.

    python train.py                                  # defaults: ResNet-18, FER+ labels
    python train.py --model alexnet --epochs 30      # the AffectNet paper's network
    python train.py --labels fer2013 --class-weighting none
    python train.py --help                           # every option

A checkpoint is saved after every epoch in runs/<run name>/. If training stops (e.g. Colab disconnects),
run the same command again and it continues where it stopped.
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch

import evaluate
import plots
from augment import augment, batches, mix_batch, prepare, to_device
from config import get_config, run_name
from data import load_data
from models import build_model
from utils import class_weights, get_device, loss_fn, predict, scores, set_seed


def main(argv=None):
    cfg = get_config(argv)
    set_seed(cfg.seed)
    device = get_device()

    data, class_names = load_data(cfg)          # may switch cfg.labels for the image-folder version
    cfg.run_dir = cfg.run_dir.parent / run_name(cfg)   # name may change if labels fell back to fer2013
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run folder:", run_dir)
    num_classes = len(class_names)

    # ---- look at the data (saved as pictures) ----
    train_counts = data["train"][1].sum(0)
    plots.class_counts(train_counts, class_names, run_dir / "plots" / "class_counts.png")
    plots.sample_grid(data["train"][0], data["train"][1].argmax(1), class_names, run_dir / "plots" / "samples.png")
    gpu = to_device(data, device)
    x_train, y_train = gpu["train"]
    demo = augment(x_train[:16].float() / 255.0).cpu().numpy()[:, 0]
    plots.image_row(demo, "Augmented training images", run_dir / "plots" / "augmented.png")

    # ---- model, weighted loss, optimizer ----
    model = build_model(cfg.model, num_classes, cfg.pretrained).to(device)
    print(f"{cfg.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    weight = class_weights(train_counts, cfg.class_weighting, device)
    print(pd.DataFrame({"images": train_counts.round().astype(int), "loss weight": weight.cpu().numpy().round(2)},
                       index=class_names))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = -(-len(x_train) // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.15)
    # EMA: a running average of the weights, used for validation, the saved checkpoint and testing
    ema_model = copy.deepcopy(model).eval() if cfg.ema > 0 else None
    for p in (ema_model.parameters() if ema_model else []):
        p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(step):
        decay = min(cfg.ema, (1 + step) / (10 + step))      # warm-up: early on, follow the model closely
        for e, m in zip(ema_model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(m.detach(), alpha=1 - decay)
            else:
                e.copy_(m)                                   # e.g. BatchNorm's num_batches_tracked

    use_amp = device.type == "cuda"             # mixed precision roughly doubles speed on a T4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume if a checkpoint exists ----
    start_epoch, best_score, history = 0, -1.0, []
    last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema_model is not None and "ema" in ck:
            ema_model.load_state_dict(ck["ema"])
        start_epoch, best_score, history = ck["epoch"] + 1, ck["best_score"], ck["history"]
        print(f"Resuming from epoch {start_epoch} (best validation {cfg.select_metric} so far: {best_score:.4f})")

    saved_cfg = {k: str(v) if k in ("data_dir", "run_dir") else v for k, v in vars(cfg).items()}
    with open(run_dir / "config.json", "w") as f:
        json.dump(saved_cfg, f, indent=2)

    # ---- training loop ----
    x_val, y_val = gpu["val"]
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, total_loss, n_seen = time.time(), 0.0, 0
        for xb, yb in batches(x_train, y_train, cfg.batch_size, shuffle=True):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                xin = prepare(xb, cfg.img_size, train=True, use_augment=cfg.augment)
                xin, yb = mix_batch(xin, yb, cfg.mixup, cfg.cutmix, cfg.mix_prob)
                logits = model(xin)
                loss = loss_fn(logits, yb, weight, cfg.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_model is not None:
                update_ema(scheduler.last_epoch)
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)

        eval_model = ema_model if ema_model is not None else model
        val_probs, val_labels, val_loss = predict(eval_model, x_val, y_val, cfg.img_size, cfg.tta,
                                                  weight=weight, label_smoothing=cfg.label_smoothing)
        val = scores(val_probs, val_labels)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / n_seen, "val_loss": val_loss, **val})
        improved = val[cfg.select_metric] > best_score
        if improved:
            best_score = val[cfg.select_metric]
            torch.save({"model": eval_model.state_dict(), "class_names": class_names, "config": saved_cfg,
                        "epoch": epoch + 1, "val": val}, best_ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_score": best_score, "history": history,
                    **({"ema": ema_model.state_dict()} if ema_model is not None else {})}, last_ckpt)
        print(f"epoch {epoch + 1:2d}/{cfg.epochs} | train loss {total_loss / n_seen:.3f} | val loss {val_loss:.3f} | "
              f"val acc {val['accuracy']:.3f} | val macro-F1 {val['macro_f1']:.3f} | {time.time() - t0:.0f}s"
              + ("  <- best" if improved else ""))

    hist = pd.DataFrame(history)
    if len(hist):
        hist.to_csv(run_dir / "history.csv", index=False)
        plots.learning_curves(hist, run_dir / "plots" / "learning_curves.png")

    # ---- final test-set evaluation with the best checkpoint ----
    evaluate.main(["--checkpoint", str(best_ckpt), "--data-dir", str(cfg.data_dir)]
                  + ([] if cfg.tta else ["--no-tta"]))


if __name__ == "__main__":
    main()
