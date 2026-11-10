"""The networks.

- alexnet:        the network used in the AffectNet paper
- resnet18:       a newer, smaller and usually more accurate network (default)
- resnet18_small: ResNet-18 adapted to native 48x48 faces (Exercise 2): 3x3 stride-1 first conv, no max-pool
- facenet:        Inception-ResNet-v1 pretrained on VGGFace2 face *recognition* (Exercise 4d)
- small_cnn:      a compact CNN trained from scratch, to show how much ImageNet pretraining helps

In each case the original classifier is replaced with a new layer that outputs one score per expression.
"""
import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FaceNetFER(nn.Module):
    """Inception-ResNet-v1 trained on VGGFace2 (3.3M face photos of 9k identities), new 8-way head.

    The data pipeline normalizes images the ImageNet way; FaceNet expects (pixel - 127.5) / 128,
    so this wrapper converts the input before passing it on.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2" if pretrained else None,
                                     classify=True, num_classes=num_classes, dropout_prob=0.4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        pixels01 = x * self.std + self.mean                  # undo ImageNet normalization -> [0, 1]
        return self.net((pixels01 * 255.0 - 127.5) / 128.0)  # FaceNet's own normalization


def build_model(name, num_classes, pretrained=True):
    tvm = torchvision.models
    if name in ("resnet18", "resnet18_small"):
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if name == "resnet18_small":
            # The standard stem (7x7 stride-2 conv + max-pool) shrinks the image 4x straight away:
            # a 48x48 face would become 12x12 before the first residual block and end as a 2x2 map.
            # A 3x3 stride-1 conv without max-pool keeps 48x48 into layer1 (final map 6x6).
            old = model.conv1.weight.data                    # (64, 3, 7, 7)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if pretrained:                                   # reuse the centre 3x3 of the pretrained 7x7 filters
                model.conv1.weight.data.copy_(old[:, :, 2:5, 2:5])
            model.maxpool = nn.Identity()
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif name == "alexnet":
        model = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "facenet":
        model = FaceNetFER(num_classes, pretrained)
    elif name == "small_cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        model = nn.Sequential(block(3, 64), block(64, 128), block(128, 256), block(256, 512),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
