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


if __name__ == "__main__":

    DATASET_SIHDR = "si-hdr"
    DATASET_KORSHUNOV = "korshunov"
    DATASET_NARWARIA = "narwaria"

    p_sihdr = "I:/Datasets/SI-HDR/reference"
    p_korshunov = "I:/Datasets/UPIQ/images/korshunov"
    p_narwaria = "I:/Datasets/UPIQ/images/narwaria"

    sihdr_images = os.listdir(p_sihdr)

    def get_image(dataset, i):
        if dataset == DATASET_SIHDR:
            im_name = sihdr_images[i]
            p = f"{p_sihdr}/{im_name}"
        elif dataset == DATASET_KORSHUNOV:
            i += 1
            p = f"{p_korshunov}/{i:0>2}/i{i:0>2}.exr"
        elif dataset == DATASET_NARWARIA:
            i += 1
            p = f"{p_narwaria}/{i:0>2}/i{i:0>2}.exr"
        else:
            raise ValueError()

        img = imread(p, True)
        ainfo(f"{dataset}-{i} ({p})", img)

        if dataset == DATASET_SIHDR:
            img = (img / 65536) ** (1/1.2) * 4000

        elif dataset == DATASET_NARWARIA:
            img = img - img.min()

        return img

    def get_images(dataset, count):
        imgs = []
        for i in range(count):
            imgs.append(get_image(dataset, i))
        return imgs

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable use of OpenEXR; must be set before 'import cv2'
    import cv2

    num_images = 10

    import numpy as np
    from matplotlib import pyplot as plt

    def plt_hist_c(dataset, num_imgs):
        imgs = get_images(dataset, num_imgs)
        channel = 0
        nbins = 250
        imgs = [img[..., channel].flatten() for img in imgs]
        imgs = np.concatenate(imgs).flatten()
        imgs = imgs + 1.0  # +1 for log scale
        logbins = np.logspace(np.log10(imgs.min()), np.log10(imgs.max()), nbins)
        hist, bins = np.histogram(imgs, bins=logbins)
        plt.stairs(hist / hist.max(), bins, alpha=0.5, label=dataset, fill=True)

    # Plot histogram
    plt_hist_c(DATASET_SIHDR, 20)
    plt_hist_c(DATASET_KORSHUNOV, 20)
    plt_hist_c(DATASET_NARWARIA, 10)
    plt.legend()
    plt.xscale('log')
    plt.xlim(1, 10000)
    # Add labels and title

    # Show plot
    plt.show()

