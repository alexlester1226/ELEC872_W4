# Facial expression recognition: the AffectNet baseline, trained on FER2013

These scripts reproduce the **classification baseline from the AffectNet paper**
(Mollahosseini, Hasani & Mahoor, *IEEE Transactions on Affective Computing*, 2017)
on the freely available **FER2013** dataset.

**Kept from the paper:** an ImageNet-pretrained CNN fine-tuned to classify expressions (the paper used AlexNet),
a **class-weighted loss** for the imbalanced classes, **8 expression classes** (with FER+ labels),
and evaluation with accuracy and macro-F1.
**Not possible here:** AffectNet's valence/arousal prediction. FER2013 has no valence/arousal labels.

## Files

| File | What it does |
|---|---|
| `config.py` | all settings and their command-line options |
| `data.py` | finds / downloads FER2013 and the FER+ labels, loads the train / validation / test splits |
| `augment.py` | GPU data pipeline: batching, augmentation, resizing, normalization |
| `models.py` | AlexNet, ResNet-18 and a small CNN |
| `utils.py` | weighted loss, prediction, accuracy / macro-F1 |
| `plots.py` | all charts (saved as PNG files) |
| `train.py` | **trains a model**, then runs the test evaluation |
| `evaluate.py` | evaluates a saved model on the test set |
| `predict.py` | predicts the expression in your own photos |

## 1. Get the data

**FER2013: nothing to do.** If `data/` is empty, `train.py` downloads the original `fer2013.tar.gz` (about 92 MB)
from a public Hugging Face mirror (the one R's torchvision package uses) and checks it against two published checksums.
When it prints *"verified: identical to the official Kaggle fer2013.csv"*, you have exactly the official file.
(The original Kaggle competition page no longer offers the files for download.)

If the automatic download fails, put a copy of FER2013 in `data/` yourself, either `fer2013.csv`
or the Kaggle dataset *msambare/fer2013* (image folders, zipped or not). The scripts find it there.

**FER+ labels** are downloaded automatically from Microsoft's FERPlus GitHub repository.
They only work with the **CSV** version of FER2013; with image folders the scripts use the original labels.

## 2. Run on Google Colab (free GPU)

*Runtime → Change runtime type → T4 GPU*, then in notebook cells:

```python
# upload the project folder (or clone it), then:
%cd fer_project

# optional: keep results in Google Drive so a disconnect doesn't lose them
from google.colab import drive
drive.mount('/content/drive')

!python train.py --out-dir /content/drive/MyDrive/fer_runs
```

If Colab disconnects, run the same command again: training continues from the last saved epoch.
A full run with the defaults should take roughly 10–20 minutes on a T4.

To see the charts in Colab:

```python
from IPython.display import Image, display
display(Image('/content/drive/MyDrive/fer_runs/resnet18_ferplus_majority_128px_sqrt_inverse/plots/confusion_matrix.png'))
```

## 3. Run on your own computer

```bash
pip install -r requirements.txt
python train.py
```

It runs on a CPU, but slowly; a GPU is strongly recommended.

## Commands

```bash
python train.py --help                               # every option, with explanations
python train.py                                      # ResNet-18, FER+ labels (default)
python train.py --model alexnet                      # the AffectNet paper's network
python train.py --labels fer2013                     # original 7-class labels
python evaluate.py --checkpoint runs/<run name>/best.pt
python predict.py --checkpoint runs/<run name>/best.pt my_photo.jpg
```

Each run gets its own folder, `runs/<model>_<labels>_<size>px_<weighting>/`, containing:
`best.pt` (best model), `last.pt` (for resuming), `config.json`, `history.csv`, `test_results.json`, and `plots/`
(class counts, sample images, augmented images, learning curves, confusion matrix, most confident mistakes).

Typical results (a rough guide): about **68–72%** test accuracy with the original FER2013 labels,
about **83–87%** with FER+ labels.

## Exercises

Change one option, train again, and compare the `test_results.json` files.

1. **Pretraining:** `--model small_cnn` vs `--model resnet18`. How much does starting from ImageNet help?
2. **The paper's network:** `--model alexnet`. How does the 2012 network compare with ResNet-18?
3. **Class imbalance:** `--class-weighting none`, `sqrt_inverse`, `inverse`. Compare accuracy *and* macro-F1,
   and the "disgust"/"contempt" rows of the confusion matrix. Why can accuracy fall while macro-F1 rises?
4. **Label quality:** `--labels fer2013` vs `--labels ferplus`. (The test sets differ slightly, since FER+ drops some images.)
5. **Soft labels:** `--ferplus-mode soft` trains on how many annotators chose each emotion,
   the same idea behind the AffectNet+ dataset (2024).
6. **Augmentation:** `--no-augment`. What happens to the gap between training and validation loss?
7. **Resolution:** `--img-size 64`, `128`, `224`. Is upscaling 48-pixel images worth the extra computation?
8. **Discussion:** which expressions get confused most? Are some "mistakes" actually wrong labels?
   What could go wrong if a model like this were used to judge people (bias, context, culture, consent)?

## Additions for the ELEC 872 workshop exercises

```bash
python train.py --model resnet18_small --img-size 48                         # Ex 2: ResNet-18 on native 48x48 faces
python train.py --epochs 60 --mixup 0.2 --cutmix 1.0 --ema 0.999             # Ex 3: longer + mixup/CutMix + EMA
python train.py --no-pretrained                                             # Ex 4a: no pretraining
python train.py --model facenet --img-size 160                              # Ex 4d: VGGFace2 face-recognition weights
```
`--model facenet` needs `pip install --no-deps facenet-pytorch`. Run folder names get suffixes
(`_scratch`, `_e60`, `_mixup0.2`, `_ema0.999`, `--tag ...`) so experiments never resume from each other's checkpoints.
