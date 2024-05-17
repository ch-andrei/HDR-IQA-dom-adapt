import torch
import torch.nn as nn
import numpy as np

from data.utils import get_imagenet_transfor_params
from modules.display_simulation.display_model import DisplayModel, DM_TYPE_SRGB, DM_TYPE_SRGB_SIMPLE, DM_TYPE_RGB_LINEAR
from modules.display_simulation.pu21.pu21_transform import PUTransform, PU21_TYPE_BANDING, PU21_TYPE_BANDING_GLARE

import torchvision.transforms.functional as functional

from utils.logging import log_warn


class PUTransformWrapper(nn.Module):
    def __init__(self,
                 normalize_pu=True,
                 normalize_pu_range_srgb=True,
                 normalize_mean_std=True,
                 normalize_mean_std_imagenet=False,
                 ):
        super().__init__()
        self.pu = PUTransform(
            encoding_type=PU21_TYPE_BANDING_GLARE, normalize=normalize_pu, normalize_range_srgb=normalize_pu_range_srgb)

        if not normalize_pu and normalize_mean_std:
            log_warn(
                "normalize_pu=False but normalize_mean_std=True",
                tag="PUTransformWrapper")

        self.normalize_mean_std = normalize_mean_std
        self.normalize_mean_std_imagenet = normalize_mean_std_imagenet
        if normalize_mean_std_imagenet:
            self.normalize_mean, self.normalize_std = get_imagenet_transfor_params()
        else:
            self.normalize_mean, self.normalize_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    def forward_pu(self, x):
        x = self.pu(x)
        if self.normalize_mean_std:
            x = functional.normalize(x, self.normalize_mean, self.normalize_std)
        return x

    def forward(self, x):
        return self.forward_pu(x)


class PuDisplayWrapper(PUTransformWrapper):
    def __init__(self,
                 display_L_max=1000,  # cd/m2
                 display_L_min=1,  # cd/m2
                 display_L_cr=1,  # cd/m2
                 display_reflectivity=0.01,
                 display_model_type=DM_TYPE_SRGB_SIMPLE,
                 **kwargs
                 ):
        """
        :param normalize: apply normalization after PU encoding. Normalize from range [0, 1] to [-1, 1].
        :param display_L_max: display maximum white point luminance cd/m2
        :param display_L_min: display minimum white point luminance cd/m2
        :param display_L_cr: display contrast ratio (controls L_blk)
        :param display_reflectivity: display reflectivity (%)
        """
        super().__init__(**kwargs)

        self.display_L_max = display_L_max
        self.display_L_min = display_L_min
        self.display_L_cr = display_L_cr

        self.dm = DisplayModel(
            L_max=display_L_max,
            L_min=display_L_min,
            L_cr=display_L_cr,
            reflectivity=display_reflectivity,
            dm_type=display_model_type
        )

    def dm_pu(self, x, E_amb=None):
        x = self.dm.forward(x, E_amb=E_amb)
        x = self.forward_pu(x)  # apply PU-encoding
        return x

    def forward(self, x, E_amb=None):
        raise NotImplementedError()


class PuDisplayWrapperRandomized(PuDisplayWrapper):
    def __init__(self,
                 randomize_ambient=True,
                 rand_distrib_normal=True,
                 rand_distrib_ambient_mean=100.,  # random ambient (lux) mean
                 rand_distrib_ambient_delta=25.,  # random ambient (lux) delta
                 randomize_L_max=True,
                 rand_L_max_delta=15,  # random ratio to reduce display Lmax
                 **kwargs
                 ):
        """
        :param randomize_L_max:
            toggle to randomize display model Lmax on each call
        :param randomize_ambient:
            toggle to randomize ambient condition on each call
        :param rand_distrib_normal:
            random distribution type, normal or uniform
        :param rand_distrib_ambient_mean:
            mean value for ambient condition (lux)
        :param rand_distrib_ambient_delta:
            deviation from mean ambient condition.
            When rand_distrib_normal is True, rand_ambient_delta approximates 1 sigma variance,
                                       False, rand_ambient_delta is the maximum deviation from mean.
        """
        self.randomize_ambient = randomize_ambient
        self.rand_distrib_normal = rand_distrib_normal

        self.rand_distrib_ambient_mean = float(rand_distrib_ambient_mean)
        self.rand_distrib_ambient_delta = float(rand_distrib_ambient_delta)

        self.randomize_L_max = randomize_L_max
        self.rand_L_max_delta = rand_L_max_delta

        self.rand_func = PuDisplayWrapperRandomized.rand_func_normal if rand_distrib_normal else \
                PuDisplayWrapperRandomized.rand_func_uniform

        super().__init__(**kwargs)

    @staticmethod
    def rand_func_normal():
        return float(np.random.randn(1))  # random value from normal distribution with mean=0, sigma=1

    @staticmethod
    def rand_func_uniform():
        return float((2. * np.random.rand(1) - 1.))  # random value in [-1, 1]

    def forward(self, x, _=None):
        if self.randomize_ambient:
            E_amb = self.rand_distrib_ambient_mean + self.rand_distrib_ambient_delta * self.rand_func()
        else:
            E_amb = self.rand_distrib_ambient_mean  # constant
        E_amb = max(0., E_amb)

        if self.randomize_L_max:
            # randomize display's maximum luminance
            L_max = self.display_L_max + self.rand_L_max_delta * self.rand_func()
            self.dm.L_max = float(L_max)

        x = self.dm_pu(x, E_amb)

        return x
