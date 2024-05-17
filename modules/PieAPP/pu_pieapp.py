import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.PieAPP.vgg2pieapp import vgg_features_to_pieapp
from modules.utils import set_grad
from utils.logging import log_warn
from utils.misc.miscelaneous import recursive_dict_flatten


# from https://github.com/gfxdisp/pu_pieapp
# AC: added use of pretrained weights from VGG16
class PUPieAPP(nn.Module):
    def __init__(self, pretrained=True, return_features=False, pretrained_vgg=True):
        nn.Module.__init__(self)
        self.extractor = FeatureExtractor()
        self.comparitor = Comparitor()

        self.return_features = return_features
        self.pretrained_vgg = pretrained_vgg
        if pretrained:
            if pretrained_vgg:
                log_warn(f"PieAPP using pretrained feature extractor (VGG16).")
                # NOTE: we use pretrained weights for VGG, but there is not perfect match between VGG and PieAPP
                # VGG is trained on Imagenet with custom normalization params (mean and std), we ignore this
                state_dict = vgg_features_to_pieapp()
                self.load_state_dict(state_dict, strict=False)  # strict=False to only load feature extractor weights

            else:
                model_path = './modules/PerceptualImageError/weights/pupieapp_weights.pt'
                log_warn(f"PU-PieAPP loading pretrained weights from {model_path}.")
                state_dict = torch.load(model_path, map_location='cuda')
                state_dict = recursive_dict_flatten(state_dict)
                self.load_state_dict(state_dict)

    def forward(self, patches):
        image_ref_patches, image_dist_patches = patches
        return self.compute_score(image_ref_patches, image_dist_patches)

    def compute_score(self, image_ref_patches, image_dist_patches):
        B, N, C, P, P = image_ref_patches.shape  # batch and patch dims

        image_ref_patches = image_ref_patches.view(B * N, C, P, P)
        image_dist_patches = image_dist_patches.view(B * N, C, P, P)

        f1, c1 = self.extractor(image_ref_patches)
        f2, c2 = self.extractor(image_dist_patches)
        scores, weights = self.comparitor(f1, c1, f2, c2)

        scores = scores.view(B, 1, N)
        weights = weights.view(B, 1, N)
        qs = (scores * weights).sum(2) / (weights.sum(2))
        qs = qs.flatten()

        return qs, (c1, c2, c1 - c2) if self.return_features else None

    def set_freeze_state(self, freeze_state, freeze_dict):
        print("PU-PieAPP: Setting freeze state to", freeze_state)

        requires_grad = not freeze_state

        if freeze_dict["freeze_feature_extractor"]:
            set_grad(self.extractor, requires_grad)

        if freeze_dict["freeze_comparitor"]:
            set_grad(self.comparitor, requires_grad)


class Func(nn.Module):
    def __init__(self, functional):
        nn.Module.__init__(self)
        self.functional = functional

    def forward(self, *input):
        return self.functional(*input)


class FeatureExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool10 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, input):
        """
        if the input
        """
        # print("\tIn Model: input size", input.size())
        # conv1 -> relu -> conv2 -> relu -> pool2 -> conv3 -> relu
        x3 = f.relu(self.conv3(self.pool2(f.relu(self.conv2(f.relu(self.conv1(input)))))))
        # conv4 -> relu -> pool4 -> conv5 -> relu
        x5 = f.relu(self.conv5(self.pool4(f.relu(self.conv4(x3)))))
        # conv6 -> relu -> pool6 -> conv7 -> relu
        x7 = f.relu(self.conv7(self.pool6(f.relu(self.conv6(x5)))))
        # conv8 -> relu -> pool8 -> conv9 -> relu
        x9 = f.relu(self.conv9(self.pool8(f.relu(self.conv8(x7)))))
        # conv10 -> relu -> pool10 -> conv11 -> relU
        x11 = f.relu(self.conv11(self.pool10(f.relu(self.conv10(x9)))))
        return torch.cat((
            self.flatten(x3),
            self.flatten(x5),
            self.flatten(x7),
            self.flatten(x9),
            self.flatten(x11),
        ), 1), x11.view(x11.size(0), -1)

    def flatten(self, x):
        """
        change vector from BxCxHxW to BxCHW
        """
        B, C, H, W = x.size()
        return x.view(B, C * H * W)


class Comparitor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fcs = nn.Sequential(
            nn.Linear(120832, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            Func(lambda x: x * 1e-2),
            nn.Linear(1, 1))
        self.weights = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
            Func(lambda x: x + 1e-6),
        )

    def forward(self, featureA, coarseA, featureRef, coarseRef):
        scores = self.fcs(featureRef - featureA)
        weights = self.weights(coarseRef - coarseA)
        return scores, weights


if __name__ == "__main__":
    model = PUPieAPP()
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.float32)
    model = model.to(device=device, dtype=torch.float32)
    B = 8
    N = 128
    P = 64
    C = 3
    data = torch.rand((B, N, C, P, P)).to(device=device, dtype=torch.float32)
    out = model((data, data))[0]
    print(out.shape)