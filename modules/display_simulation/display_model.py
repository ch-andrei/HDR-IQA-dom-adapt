import torch
import torch.nn as nn
import numpy as np

from utils.logging import log_warn
from utils.misc.timer import Timer


DM_TYPE_SRGB = 0
DM_TYPE_SRGB_SIMPLE = 1
DM_TYPE_RGB_LINEAR = 2

DM_TYPES = {
    DM_TYPE_SRGB: "SRGB",
    DM_TYPE_SRGB_SIMPLE: "SRGB_SIMPLE",
    DM_TYPE_RGB_LINEAR: "RGB_LINEAR",
}


# from https://github.com/pytorch/pytorch/issues/50334
def torch_interp(x, xp, fp):
    """
    functionality of np.interp but in torch
    :param x: input
    :param xp: x values for interp
    :param fp: f(x) values for interp
    :return:
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x + b[indices]


def srgb2rgb(srgb: torch.Tensor):
    """
        sources:
        https://en.wikipedia.org/wiki/Relative_luminance
        https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016perceptual_display.pdf
        converts from sRGB to linear RGB
    :param srgb:
    :param gamma:
    :return:
    """
    c = 0.04045
    a = 0.055
    gamma = 2.4
    return torch.lerp(srgb / 12.92, ((srgb + a) / (1.0 + a)) ** gamma, (c < srgb).float())


def dm_srgb(Vsrgb, L_max, L_blk):
    rgb = srgb2rgb(Vsrgb)
    return (L_max - L_blk) * rgb + L_blk


def dm_srgb_simple(Vsrgb, L_max, L_blk, gamma):
    return (L_max - L_blk) * torch.pow(Vsrgb, gamma) + L_blk


def dm_rgb_linear(Vrgb, L_max, L_blk):
    return (L_max - L_blk) * Vrgb + L_blk


class DisplayModel(nn.Module):
    __E_AMB_MAX = 25000

    @property
    def max_E_amb(self):
        return float(self.__E_AMB_MAX)

    def dm_type_name(self):
        return DM_TYPES[self.dm_type]

    def __init__(self,
                 L_max=100,
                 L_min=5,
                 L_cr=1000,
                 gamma=2.2,
                 reflectivity=0.01,
                 dm_type=DM_TYPE_SRGB_SIMPLE,
                 ):
        """
        :param L_max:
            the highest display luminance value for a white pixel.
        :param L_min:
            the lowest display luminance value for a white pixel.
        :param L_cr:
            display contrast ratio, used to compute L_blk, the display luminance of the black level.
            L_blk = L_max / L_cr
        :param gamma:
            display gamma
        :param reflectivity:
            default display reflectivity ratio (this depends on display surface)
            Examples:
            1. ITU-R BT.500-11 recommends 6% or 0.06 for common displays
            https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.500-11-200206-S!!PDF-E.pdf
            2. HDR-VDP (Rafal Mantiuk) uses 1% or 0.01
        :param dm_type: [DM_TYPE_SRGB, DM_TYPE_SRGB_SIMPLE, DM_TYPE_RGB_LINEAR]
            selects display model type.
            DM_TYPE_SRGB converts sRGB to linear RGB then applies DM equation
            DM_TYPE_SRGB_SIMPLE applies gamma correction to sRGB without converting to linear RGB (simpler) then DM eq.
            DM_TYPE_RGB_LINEAR assumes inputs are already in linear color and only applies DM
        """

        super().__init__()

        self.L_max = float(L_max)
        self.L_min = float(L_min)
        self.L_cr = L_cr
        self.gamma = gamma
        self.reflectivity = reflectivity

        self.dm_type = dm_type
        if dm_type not in [DM_TYPE_SRGB, DM_TYPE_SRGB_SIMPLE, DM_TYPE_RGB_LINEAR]:
            raise NotImplementedError(f"Unsupported Display Model type {dm_type}")
        log_warn(f"DisplayModel with type={DM_TYPES[dm_type]}.")

    def __get_E_amb(self, E_amb):
        if isinstance(E_amb, torch.Tensor):
            return torch.clamp(E_amb, 0., self.max_E_amb)
        else:
            return np.clip(E_amb, 0, self.max_E_amb)

    def compute_display_response(self, V):
        L_max = self.L_max
        L_blk = self.L_max / self.L_cr

        if self.dm_type == DM_TYPE_SRGB:
            return dm_srgb(V, L_max, L_blk)
        elif self.dm_type == DM_TYPE_SRGB_SIMPLE:
            return dm_srgb_simple(V, L_max, L_blk, self.gamma)
        elif self.dm_type == DM_TYPE_RGB_LINEAR:
            return dm_rgb_linear(V, L_max, L_blk)
        else:
            raise ValueError("Unsupported display model type.")

    def forward(
            self,
            V,  # input values in range [0-1] (luma or linear RGB channels)
            E_amb=None,  # ambient illuminance in lux (single float or tensor)
    ):
        if E_amb is None:
            E_amb = 0

        else:
            E_amb = self.__get_E_amb(E_amb)

            if isinstance(E_amb, torch.Tensor):
                # match the input size as K x 1 x 1 x 1 x 1
                E_amb = E_amb.view(-1, *[1 for _ in range(len(V.shape) - 1)])

        L_d = self.compute_display_response(V)
        L_refl = self.reflectivity * E_amb / np.pi

        return L_d + L_refl
