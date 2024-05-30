import numpy as np
from data.patch_datasets import PatchFRIQADataset


class TID2013Dataset(PatchFRIQADataset):
    num_ref_images = 25
    num_dist_images = 120
    num_distortions = 24
    img_dim = (384, 512)

    def __init__(self,
                 name="TID2013",
                 path="tid2013",
                 **kwargs
                 ):
        """
        The original TID2013 dataset provides 3000 quality comparisons for 25 references images (512x384 resolution).

        :param path: path to TID2013 directory
        :param patch_dim: int or tuple (h x w), will sample a random square patch of this size
        :param patch_count: number of samples to return for each image

        """
        super(TID2013Dataset, self).__init__(
            name=name,
            path=path,
            # From TID2013 readme: "Higher value of MOS (0 - minimal, 9 - maximal) corresponds to higher visual
            # quality of the image."; reversing the scores is required as we assume zero is perfect quality.
            # reverse False before linearize, reverse True after linearize to get the desired ordering.
            qs_reverse=True,
            qs_linearize=True,
            **kwargs
        )

    def read_dataset(
            self,
            # NOTE: params are exposed, because KADID reuses this read function with different params
            reference_images_path="/reference_images",
            distorted_images_path="/distorted_images",
            q_file_name="mos_with_names.txt",
            split_char=" ",
            q_ind=0,
            filename_ind=1,
            filename_ext="bmp",
            has_header=False
    ):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is MOS for original TID2013 or JOD for TID2013+.
        :return:
        """
        reference_images_path = self.path + reference_images_path
        distorted_images_path = self.path + distorted_images_path

        paths_ref, paths_dist, qs = [], [], []
        q_file_path = self.path + "/" + q_file_name
        with open(q_file_path, 'r') as q_file:
            if has_header:
                q_file.__next__()  # skip header line

            for line in q_file:
                line = line.strip().split(split_char)  # split by comma or space

                # the first 3 letters are the reference file name
                path_reference = reference_images_path + '/' + line[filename_ind][0:3] + "." + filename_ext
                path_distorted = distorted_images_path + '/' + line[filename_ind]
                q = float(line[q_ind])

                paths_ref.append(path_reference)
                paths_dist.append(path_distorted)
                qs.append(q)

        dist_images_per_image = [self.num_dist_images for _ in range(self.num_ref_images)]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)


class TID2008Dataset(TID2013Dataset):
    # num_ref_images inherited from TID2013Dataset
    num_dist_images = 68
    num_distortions = 17

    def __init__(self,
                 **kwargs):
        super(TID2008Dataset, self).__init__(path='tid2008', name="TID2008", **kwargs)
