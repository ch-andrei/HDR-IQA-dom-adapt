import torch
import torch.nn as nn

from utils.logging import log_warn

PU21_TYPE_BANDING = 0
PU21_TYPE_BANDING_GLARE = 1
PU21_TYPE_PEAKS = 2
PU21_TYPE_PEAKS_GLARE = 3


class PUTransform(nn.Module):
    """
    Transform absolute linear luminance values to/from the perceptually
    uniform (PU) space. This class is intended for adapting image quality
    metrics to operate on HDR content.

    The derivation of the PU21 encoding is explained in the paper:

    R. K. Mantiuk and M. Azimi, "PU21: A novel perceptually uniform encoding for adapting existing quality metrics
    for HDR," 2021 Picture Coding Symposium (PCS), 2021, pp. 1-5, doi: 10.1109/PCS50896.2021.9477471.

    Aydin TO, Mantiuk R, Seidel H-P.
    Extending quality metrics to full luminance range images.
    In: Human Vision and Electronic Imaging. Spie 2008. no. 68060B.
    DOI: 10.1117/12.765095

    The original MATLAB implementation is ported to Python and modified for realistic display systems.
    """

    # PU21 parameters in order: [PU21_TYPE_BANDING, PU21_TYPE_BANDING_GLARE, PU21_TYPE_PEAKS, PU21_TYPE_PEAKS_GLARE]
    __par = [
        [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484],
        [0.353487901, 0.3734658629, 8.2770492e-05, 0.9062562627, 0.0915030316, 0.90995172, 596.3148142],
        [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577],
        [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374],
    ]

    def __init__(self,
                 encoding_type: int = PU21_TYPE_BANDING_GLARE,
                 normalize=True,
                 normalize_range_srgb=True,
                 L_min_0=False,
                 ):
        """
        :param encoding_type:
        :param normalize: toggle for rescaling the output to range ~[0.0, 1.0]
        :param normalize_range_srgb: when true, PU-encoded value 255 (instead of P_max) will map to 1.0.
            Output range then is not ~[0.0, 1.0] but ~[0.0, 2.5] with sRGB@100cd/m2 encoded as approximately [0.0, 1.0]
        :param L_min_0: extend the minimum L range to 0.0cd/m2 instead of 0.005cd/m2
        """
        super().__init__()

        self.L_min = 0.0 if L_min_0 else 0.005
        self.L_max = 10000

        if encoding_type not in [PU21_TYPE_BANDING, PU21_TYPE_BANDING_GLARE, PU21_TYPE_PEAKS, PU21_TYPE_PEAKS_GLARE]:
            raise ValueError("Unsupported PU21 encoding type.")

        self.encoding_type = encoding_type
        self.par = self.__par[encoding_type]

        pu_encode = lambda x: self.pu_encode(torch.as_tensor(x)).item()  # for float inputs

        # NOTE: start with normalization/L_min_0 disabled to compute scaling parameters using the original PU scale
        self.L_min_0 = False
        self.normalize = False
        if L_min_0:
            # compute slope at L=0.005cd/m2 and the corresponding y-intersect given linear equation of form y=ax+b
            h = 1e-4  # Note: when using autocast float16, will lose precision for small h (h=~1e-4 works well)
            L_005 = 0.005
            self.P_005_m = (pu_encode(L_005 + h) - pu_encode(L_005)) / h  # slope
            self.P_005_y = pu_encode(L_005) - L_005 * self.P_005_m  # y-intersect
        # set the final L_min_0 parameter
        self.L_min_0 = L_min_0

        # compute upper range for normalization
        # use PU-encoded value at L=10000cd/m2 or at typical CRT display luminance (SDR) of 100cd/m2
        self.P_max = pu_encode(100 if normalize_range_srgb else 10000)
        # compute lower range for normalization
        self.P_min = pu_encode(self.L_min)

        # print("P_min", self.P_min)
        # print("P_max", self.P_max)

        # set the final normalization params
        self.normalize = normalize
        self.normalize_range_srgb = normalize_range_srgb

    @staticmethod
    def pu_encode_poly(Y, p):
        return p[6] * (((p[0] + p[1] * Y ** p[3]) / (1 + p[2] * Y ** p[3])) ** p[4] - p[5])

    def pu_encode(self, Y):
        """
        Convert from linear (optical) values Y to encoded (electronic) values V
        Y should be scaled in the absolute units (nits, cd/m^2).
        """
        Y = torch.clip(Y, self.L_min, self.L_max)
        V = self.pu_encode_poly(Y, self.par)

        if self.L_min_0:
            V_005 = self.P_005_m * Y + self.P_005_y  # linear extension for 0.0 < L < 0.005 cd/m2
            V = torch.lerp(V_005, V, (0.005 < Y).float())

        return V

    def forward(self, Y):
        V = self.pu_encode(Y)

        if self.normalize or self.L_min_0:
            V -= self.P_min

        if self.normalize:
            V /= (self.P_max - self.P_min)

        return V


# from Rec. ITU-R BT.2100-2
# https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf
class PerceptualQuantizerTransform(nn.Module):
    def __init__(self, normalize=True, normalize_range_srgb=True):
        super().__init__()
        self.L = 10000
        self.m1 = 0.1593017578125
        self.m2 = 78.84375
        self.c1 = 0.8359375
        self.c2 = 18.8515625
        self.c3 = 18.6875

        self.normalize = False  # disable normalization when computing self.pq_at_100cdm2
        self.pq_at_100cdm2 = self.pq_oetf(torch.as_tensor(100))  # compute pq value at 100cd/m2

        self.normalize = normalize  # Note that PQ is by default normalize to range 0.0 - 1.0
        self.normalize_range_srgb = normalize_range_srgb

    def pq_oetf(self, FD):
        """
        this function is the PQ OETF (inverse of OOTF) which maps luminance to PQ space (output range 0-1)
        :param FD: the luminance of a displayed linear component {RD, GD, BD} or YD or ID, in cd/m2
        :return:
        """
        Y = torch.clip(FD / 10000, 0.0, 1.0)
        Ym1 = torch.pow(Y, self.m1)
        e = torch.pow((self.c1 + self.c2 * Ym1) / (1 + self.c3 * Ym1), self.m2)
        return e

    def forward(self, FD):
        e = self.pq_oetf(FD)  # PQ encoded range in 0-1

        if not self.normalize:
            e *= 256  # scale to 256 steps by default

        if self.normalize_range_srgb:
            e /= self.pq_at_100cdm2  # align 1.0 to self.pq_at_100cd/m2

        return e


# deprecated in favor of implementation directly adapted from Matlab (see above)
class PUTransform08(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.P_min = 0.31964111328125
        self.P_max = 1270.1545
        self.logL_min = -1.7647
        self.logL_max = 8.9979
        self.poly6 = [-2.76770417e-03, 5.45782779e-02, -1.31306184e-01, -4.05827702e+00,
                      3.74004810e+01, 2.84345651e+01, 5.15162034e+01]
        self.poly3 = [2.5577829, 17.73608751, 48.96952155, 45.55950728]
        self.epsilon = 1e-8

        self.normalize = normalize

    def forward(self, im):
        """
        :param im: display referred luminance (after display equation/simulation) in cd/m2
        :return:
        """
        im = self.log_clamp(im)
        im = self.apply_pu(im)
        if self.normalize:
            im = self.scale(im)
        return im

    def log_clamp(self, im):
        return torch.clamp(torch.log10(torch.clamp(im, self.epsilon, None)), self.logL_min, self.logL_max)

    def apply_pu(self, im):
        im2 = im * im  # img ** 2
        im3 = im2 * im  # img ** 3
        p3 = self.poly3
        p6 = self.poly6
        third_ord = p3[0] * im3 + p3[1] * im2 + p3[2] * im + p3[3]
        sixth_ord = p6[0] * im3 * im3 + p6[1] * im2 * im3 + \
                    p6[2] * im2 * im2 + p6[3] * im3 + \
                    p6[4] * im2 + p6[5] * im + p6[6]

        lerp_weights = (im >= 0.8).float()
        return torch.lerp(third_ord, sixth_ord, lerp_weights)

    def scale(self, x):
        """
        scale x to values between 0 and 1
        """
        return (x - self.P_min) / (self.P_max - self.P_min)
