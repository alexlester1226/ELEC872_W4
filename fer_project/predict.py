"""Predict the expression in your own photos.

    python predict.py --checkpoint runs/<run name>/best.pt photo1.jpg photo2.png

OpenCV finds the largest face, it is cropped and turned into a 48x48 grayscale image like FER2013,
and a chart with the prediction is saved next to each photo as <photo>_prediction.png.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import plots
from augment import prepare
from models import build_model
from utils import get_device


def find_face(gray):
    """Return (x, y, w, h) of the largest face, or the whole image if none is found."""
    faces = []
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not detector.empty():
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    except Exception:
        pass
    if len(faces) == 0:
        print("  no face found (or face detector unavailable), using the whole image")
        return 0, 0, gray.shape[1], gray.shape[0]
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict facial expressions in photos")
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("images", nargs="+", help="photo files")
    args = parser.parse_args(argv)

    device = get_device()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names, cfg = ck["class_names"], ck["config"]
    model = build_model(cfg["model"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    for path in args.images:
        print(path)
        img = cv2.imread(str(path))
        if img is None:
            print("  could not read this file")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = find_face(gray)
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
        xb = torch.from_numpy(face).view(1, 1, 48, 48).to(device)
        with torch.no_grad():
            xin = prepare(xb, cfg["img_size"], train=False)
            probs = ((model(xin) + model(xin.flip(3))) / 2).float().softmax(1)[0].cpu().numpy()
        for i in np.argsort(-probs)[:3]:
            print(f"  {class_names[i]:9s} {probs[i]:.1%}")
        out = Path(path).with_name(Path(path).stem + "_prediction.png")
        plots.photo_prediction(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (x, y, w, h), face, probs, class_names, out)
        print("  saved", out)


if __name__ == "__main__":
    main()
