"""Find, download and load FER2013 (CSV or image-folder version) with FER2013 or FER+ labels.

Every split is returned as
    X: uint8 images, shape (N, 1, 48, 48)
    Y: one probability vector per image, shape (N, num_classes)
       (one-hot for ordinary labels, annotator vote shares for FER+ "soft" mode)
"""
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FERPLUS_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
FERPLUS_CLASSES = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
KAGGLE_COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
KAGGLE_DATASET = "msambare/fer2013"

# Mirror of the original Kaggle fer2013.tar.gz, used by R's torchvision package (archive checksum from its source).
FER_MIRROR_URL = "https://huggingface.co/datasets/DerrickUnleashed/FER-2013/resolve/main/fer2013.tar.gz"
FER_MIRROR_MD5 = "ca95d94fe42f6ce65aaae694d18c628a"
# Checksum of the official fer2013.csv, as used by PyTorch's torchvision.datasets.FER2013.
FER_CSV_MD5 = "f8428a1edbd21e88f42c73edd2a14f95"


# ----------------------------------------------------------------------------- finding the files

def unpack_archives(root):
    """Extract every .zip / .tar.gz under root, including archives inside archives (each only once).

    The Kaggle competition download is a zip that itself contains icml_face_data.csv.zip,
    train.csv.zip, test.csv.zip and fer2013.tar.gz.
    """
    while True:
        archives = [p for pat in ("*.zip", "*.tar.gz", "*.tgz") for p in Path(root).rglob(pat)
                    if not p.with_name(p.name + ".extracted").exists()]
        if not archives:
            return
        for arch in archives:
            _extract(arch)


def _extract(arch):
    print("Extracting", arch)
    if arch.suffix == ".zip":
        with zipfile.ZipFile(arch) as z:
            z.extractall(arch.parent)
    else:
        with tarfile.open(arch) as t:
            if sys.version_info >= (3, 12):
                t.extractall(arch.parent, filter="data")
            else:
                t.extractall(arch.parent)
    arch.with_name(arch.name + ".extracted").touch()   # remember it, so it isn't extracted again


def find_fer_csv(root):
    """First CSV under root with emotion / pixels / usage columns (fer2013.csv or Kaggle's icml_face_data.csv)."""
    for path in sorted(Path(root).rglob("*.csv")):
        try:
            cols = [c.strip().lower() for c in pd.read_csv(path, nrows=1).columns]
        except Exception:
            continue
        if {"emotion", "pixels", "usage"} <= set(cols):
            return path
    return None


def find_fer_folders(root):
    """Folder that contains train/<class>/ and test/<class>/ subfolders, if any."""
    for train_dir in sorted(Path(root).rglob("train")):
        test_dir = train_dir.parent / "test"
        if train_dir.is_dir() and test_dir.is_dir() and any(p.is_dir() for p in train_dir.iterdir()):
            return train_dir.parent
    return None


def try_kaggle_download(dest):
    """Download FER2013 with the Kaggle API if a kaggle.json token can be found."""
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        for cand in [Path(dest) / "kaggle.json", Path("kaggle.json"), Path("/content/kaggle.json")]:
            if cand.exists():
                token.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(cand, token)
                token.chmod(0o600)
                break
    if not token.exists():
        return False
    for cmd in (["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(dest)],
                ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest)]):
        print("Running:", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print("  the kaggle command is not installed (pip install kaggle)")
            return False
        if result.returncode == 0:
            return True
        print("  failed:", (result.stderr or result.stdout).strip()[:300])
    return False


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def try_mirror_download(dest):
    """Download the original fer2013.tar.gz from a public mirror and check it against published checksums."""
    archive = Path(dest) / "fer2013.tar.gz"
    print(f"Downloading FER2013 (about 92 MB) from {FER_MIRROR_URL} ...")
    try:
        urllib.request.urlretrieve(FER_MIRROR_URL, archive)
    except Exception as e:
        print("  download failed:", e)
        archive.unlink(missing_ok=True)
        return False
    if md5(archive) != FER_MIRROR_MD5:
        print("  the downloaded file does not match the expected checksum; deleting it")
        archive.unlink()
        return False
    unpack_archives(dest)
    csv_path = next(Path(dest).rglob("fer2013.csv"), None)
    if csv_path is None:
        print("  the archive did not contain fer2013.csv")
        return False
    if md5(csv_path) == FER_CSV_MD5:
        print("  verified: identical to the official Kaggle fer2013.csv")
    else:
        print("  warning: fer2013.csv differs from the official file's checksum")
    return True


def locate_dataset(data_dir, allow_download=True):
    """Return ("csv", path) or ("folders", path), downloading FER2013 if it isn't there yet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    unpack_archives(data_dir)
    csv_path = find_fer_csv(data_dir)
    if csv_path:
        return "csv", csv_path
    folder = find_fer_folders(data_dir)
    if folder:
        return "folders", folder
    if allow_download and (try_mirror_download(data_dir) or try_kaggle_download(data_dir)):
        return locate_dataset(data_dir, allow_download=False)
    raise FileNotFoundError(
        f"FER2013 not found in '{data_dir}' and the automatic download failed. "
        f"Put fer2013.csv (or a FER2013 zip from Kaggle) there. See README.md.")


# ----------------------------------------------------------------------------- loading

def one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    pixels = np.stack([np.asarray(p.split(), dtype=np.uint8) for p in df["pixels"]]).reshape(-1, 48, 48)
    usage = df["usage"].astype(str).str.strip().values
    split = np.select([usage == "Training", usage == "PublicTest", usage == "PrivateTest"],
                      ["train", "val", "test"], "drop")
    return df, pixels, split


def load_ferplus_targets(data_dir, n_rows, mode):
    """FER+ labels (10 annotators per image). Returns (targets, keep-mask)."""
    path = Path(data_dir) / "fer2013new.csv"
    if not path.exists():
        print("Downloading FER+ labels ...")
        urllib.request.urlretrieve(FERPLUS_URL, path)
    fp = pd.read_csv(path)
    fp.columns = [c.strip() for c in fp.columns]
    if len(fp) != n_rows:
        raise ValueError(f"FER+ has {len(fp)} rows but the FER2013 CSV has {n_rows}; they must match row for row.")
    votes = fp[FERPLUS_COLUMNS].to_numpy(dtype=np.float32)
    other = fp[["unknown", "NF"]].to_numpy(dtype=np.float32)   # "unknown" emotion / "not a face"
    name = fp["Image name"].fillna("").astype(str).str.strip()
    top2 = np.sort(votes, axis=1)[:, -2:]
    keep = (
        (name != "").to_numpy()             # FER+ removed some images (no file name)
        & (votes.sum(1) > 0)
        & (votes.max(1) > other.max(1))     # the top choice is a real emotion
        & (top2[:, 1] > top2[:, 0])         # no tie between the two top emotions
    )
    if mode == "soft":
        targets = votes / np.clip(votes.sum(1, keepdims=True), 1, None)
    else:
        targets = one_hot(votes.argmax(1), len(FERPLUS_CLASSES))
    return targets.astype(np.float32), keep


def load_from_csv(csv_path, labels, ferplus_mode, data_dir):
    df, pixels, split = load_csv(csv_path)
    if labels == "ferplus":
        targets, keep = load_ferplus_targets(data_dir, len(df), ferplus_mode)
        class_names = FERPLUS_CLASSES
    else:
        targets = one_hot(df["emotion"].to_numpy(), len(FER2013_CLASSES))
        keep = np.ones(len(df), dtype=bool)
        class_names = FER2013_CLASSES
    data = {s: (pixels[(split == s) & keep][:, None], targets[(split == s) & keep]) for s in ["train", "val", "test"]}
    return data, class_names


def load_from_folders(root, seed):
    from PIL import Image
    root = Path(root)
    class_names = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())

    def read_dir(d):
        xs, ys = [], []
        for ci, c in enumerate(class_names):
            for f in sorted((d / c).glob("*")):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    xs.append(np.asarray(Image.open(f).convert("L").resize((48, 48))))
                    ys.append(ci)
        return np.stack(xs)[:, None], np.array(ys)

    x_tr, y_tr = read_dir(root / "train")
    x_te, y_te = read_dir(root / "test")
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=seed)
    k = len(class_names)
    data = {"train": (x_tr, one_hot(y_tr, k)), "val": (x_va, one_hot(y_va, k)), "test": (x_te, one_hot(y_te, k))}
    return data, class_names


def load_data(cfg):
    """Return (data dict, class names). May switch cfg.labels to "fer2013" for the folder version."""
    fmt, path = locate_dataset(cfg.data_dir)
    print(f"Found FER2013 ({fmt}) at {path}")
    if fmt == "csv":
        data, class_names = load_from_csv(path, cfg.labels, cfg.ferplus_mode, cfg.data_dir)
    else:
        if cfg.labels == "ferplus":
            print("FER+ labels need the CSV version of FER2013. Falling back to the original FER2013 labels.")
            cfg.labels = "fer2013"
        data, class_names = load_from_folders(path, cfg.seed)
    for s in ["train", "val", "test"]:
        print(f"  {s:5s}: {len(data[s][0]):6d} images")
    return data, class_names
