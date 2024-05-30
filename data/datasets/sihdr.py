import os

from data.patch_datasets import PatchFRIQADataset
from data.utils import imread
from modules.utils import tinfo, ainfo


class SIHDRDataset(PatchFRIQADataset):
    num_ref_images = 181
    num_dist_images = 1
    num_distortions = 1
    img_dim = (1280, 1888)

    def __init__(self,
                 **kwargs
                 ):
        """
        https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/
        This dataset contains 181 RAW exposure stacks selected to cover a wide range of image content and lighting
        conditions. Each scene is composed of 5 RAW exposures and merged into an HDR image using the estimator
        that accounts photon noise [3] (code at HDRutils). A simple color correction was applied using a
        reference white point and all merged HDR images were resized to 1920×1280 pixels.
        """
        super(SIHDRDataset, self).__init__(
            name="SIHDR",
            path="SI-HDR",
            is_hdr=True,  # dataset provides luminance

            qs_normalize=False,
            qs_reverse=False,
            qs_normalize_mean_std=False,
            qs_linearize=False,

            # use_ref_img_cache=True,

            **kwargs
        )

    def process_qs(self):
        return  # do nothing

    def is_hdr_image(self, path):
        return True  # all images are HDR in SI-HDR dataset

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is MOS for original TID2013 or JOD for TID2013+.
        :return:
        """
        images_path = f"{self.path}/reference"
        image_files = os.listdir(images_path)

        paths_ref = [f"{images_path}/{image_file}" for image_file in image_files]
        paths_dist = [f"{images_path}/{image_file}" for image_file in image_files]
        eps = 1e-6
        qs = [eps for _ in image_files]  # near zero error values
        dist_images_per_image = [1 for _ in image_files]  # there are only reference images in this dataset
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)

    def img_pretransform(self, img):
        # SI-HDR images are stored as 16-bit HDR, 2^16 = 65536
        return img / 65536.0
