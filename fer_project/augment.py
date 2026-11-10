"""Data pipeline that runs on the GPU.

Colab Free has only 2 CPU cores, so augmenting images on the CPU would leave the GPU waiting.
FER2013 is small (about 80 MB), so the whole dataset lives on the GPU and whole batches are augmented there.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_device(data, device):
    return {s: (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)) for s, (x, y) in data.items()}


def batches(x, y, batch_size, shuffle):
    order = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for i in range(0, len(x), batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]


def augment(x):
    """Random flip, rotation (±15°), zoom (±10%), shift (±8%), brightness/contrast and erasing.

    x: float tensor (B, 1, H, W) with values in [0, 1].
    """
    b, dev = x.size(0), x.device
    r = lambda: torch.rand(b, device=dev) * 2 - 1            # uniform in [-1, 1]

    # horizontal flip
    x = torch.where((torch.rand(b, device=dev) < 0.5).view(b, 1, 1, 1), x.flip(3), x)

    # rotation + zoom + shift in one affine warp
    angle, scale = r() * np.deg2rad(15), 1 + r() * 0.10
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, r() * 0.16], 1),
                         torch.stack([sin, cos, r() * 0.16], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # contrast and brightness
    mean = x.mean((2, 3), keepdim=True)
    x = ((x - mean) * (1 + r().view(b, 1, 1, 1) * 0.3) + mean) * (1 + r().view(b, 1, 1, 1) * 0.2)

    # random erasing: hide a rectangle covering 2–15% of the face in half of the images
    h, w = x.shape[2:]
    area = torch.empty(b, device=dev).uniform_(0.02, 0.15)
    aspect = torch.exp(torch.empty(b, device=dev).uniform_(np.log(0.3), np.log(3.3)))
    eh = (torch.sqrt(area * aspect) * h).clamp(1, h - 1)
    ew = (torch.sqrt(area / aspect) * w).clamp(1, w - 1)
    y0, x0 = torch.rand(b, device=dev) * (h - eh), torch.rand(b, device=dev) * (w - ew)
    rows = torch.arange(h, device=dev).view(1, h, 1)
    cols = torch.arange(w, device=dev).view(1, 1, w)
    box = ((rows >= y0.view(b, 1, 1)) & (rows < (y0 + eh).view(b, 1, 1))
           & (cols >= x0.view(b, 1, 1)) & (cols < (x0 + ew).view(b, 1, 1)))
    box &= (torch.rand(b, device=dev) < 0.5).view(b, 1, 1)
    x = x.masked_fill(box.unsqueeze(1), 0.5)
    return x.clamp(0, 1)


def prepare(x_uint8, img_size, train=False, use_augment=True):
    """uint8 grayscale batch (B,1,48,48) -> normalized 3-channel float batch at img_size."""
    x = x_uint8.float() / 255.0
    if train and use_augment:
        x = augment(x)
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def mix_batch(x, y, mixup_alpha=0.0, cutmix_alpha=0.0, prob=0.5):
    """Mixup / CutMix on a prepared batch x (B,3,H,W) with probability-vector targets y (B,K).

    mixup:  blend two whole faces, x = lam*x1 + (1-lam)*x2, and their labels the same way
    CutMix: paste a rectangle of face 2 into face 1; labels mixed by the pasted area
    """
    use = [a for a in (mixup_alpha, cutmix_alpha) if a > 0]
    if not use or np.random.rand() >= prob:
        return x, y
    perm = torch.randperm(x.size(0), device=x.device)
    do_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    if do_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        h, w = x.shape[2:]
        ch, cw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
        x0, x1 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)       # actual kept area
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * y + (1 - lam) * y[perm]
