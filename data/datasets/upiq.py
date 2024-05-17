import numpy as np

from data.patch_datasets import PatchFRIQADataset
from data.utils import normalize_values, reverse_values


class UPIQDataset(PatchFRIQADataset):
    # counts and sizes vary
    num_ref_images = None
    num_dist_images = None
    img_dim = None

    # entry format:
    # condition_id,dataset,content_id,dist_id,dist_level,is_hdr,test_file,reference_file,pix_per_deg,distortion_name,repeating_content_id,JOD
    # 0            1       2          3       4          5      6         7              8           9               10                   11

    SUBSET_NAME_LIVE = "live"
    SUBSET_NAME_TID = "tid2013"
    SUBSET_NAME_NARWARIA = "narwaria"
    SUBSET_NAME_KORSHUNOV = "korshunov"

    # equivalence between reference images in LIVE and TID datasets; -1 indicates that image is only in LIVE
    # 10 out of 29 ref images in LIVE are unique (not in TID)
    # maps images numbers from LIVE to TID (ex: "1: 5" means that LIVE image "i01.png" maps to "i05.png" in TID)
    LIVE_TID_EQUIVALENT = {
        1: 5,
        2: -1,
        3: 8,
        4: 3,
        5: -1,
        6: -1,
        7: -1,
        8: -1,
        9: -1,
        10: -1,
        11: 22,
        12: 19,
        13: 21,
        14: -1,
        15: -1,
        16: 16,
        17: 24,
        18: 23,
        19: 20,
        20: 14,
        21: 6,
        22: 9,
        23: 10,
        24: 11,
        25: 17,
        26: 13,
        27: -1,
        28: 18,
        29: 4,
    }

    JOD_min = -8.854307636
    JOD_max = 0.943130668

    @property
    def data_subsets(self):
        raise NotImplementedError()

    def __init__(self,
                 name_tag,
                 is_hdr,  # controls which subset of the dataset to build (SDR=[LIVE,TID], HDR=[Korshunov, Narwaria])
                 **kwargs
                 ):

        super(UPIQDataset, self).__init__(
            path='UPIQ',
            name=f"UPIQ-{name_tag}",
            is_hdr=is_hdr,

            qs_normalize=True,  # enable normalize
            qs_reverse=True,  # enable reverse

            **kwargs
        )

    def process_qs(self):
        print("UPIQ custom self.process_qs()...")
        print("Before processing Qs (min/mean/max):", self.qs.min(), self.qs.mean(), self.qs.max())

        qs_raw = self.qs.copy()

        # for UPIQ, only need to normalize and reverse the values
        qs = normalize_values(qs_raw, self.qs_normalize, self.qs_normalize_mean_std, self.JOD_min, self.JOD_max,
                              inplace=False)

        jod_range = (self.JOD_min, self.JOD_max)
        # 0-1 if normalize enabled, else JOD_min-JOD_max
        reverse_range = normalize_values(np.array(jod_range), self.qs_normalize, self.qs_normalize_mean_std)
        qs = reverse_values(qs, self.qs_reverse, reverse_range[0], reverse_range[1])

        self.qs = qs

        # call to original process_qs() to plot data if plotting is enabled
        self.plot_process_qs(qs_raw, self.qs, jod_range, reverse_range)

    def live2tid_ref_image(self, live_img_id):
        img_id = live_img_id
        image_id_num = int(live_img_id.replace("l-i", ""))
        tid_image_id = self.LIVE_TID_EQUIVALENT[image_id_num]
        if -1 < tid_image_id:
            # register image under TID2013 ref image equivalent, if LIVE ref image is also in TID
            img_id = f"t-i{tid_image_id:02d}"
        return img_id

    def read_dataset(self):
        """
        returns a list of tuples (reference_image_path, distorted_image_path, quality)
        where q is MOS for original TID2013 or JOD for TID2013+.
        :return:
        """
        images_path = f"{self.path}/images"
        q_file_path = f"{self.path}/upiq_subjective_scores.csv"

        unique_images = set()
        with open(q_file_path, 'r') as q_file:
            q_file.__next__()  # skip header line
            for line in q_file:
                line = line.strip().split(',')
                image_id = line[2]
                unique_images.add(image_id)

        comparisons_per_image = {image_id: [] for image_id in unique_images}

        with open(q_file_path, 'r') as q_file:
            q_file.__next__()  # skip header line
            for line in q_file:
                line = line.strip().split(',')

                dataset_name = line[1].lower()
                if dataset_name not in self.data_subsets:
                    continue

                # the first 3 letters are the reference file name
                jod = float(line[11])
                path_reference = f"{images_path}/{line[7]}"
                path_distorted = f"{images_path}/{line[6]}"

                if self.is_hdr:
                    path_reference = path_reference.replace(".png", ".exr")
                    path_distorted = path_distorted.replace(".png", ".exr")

                # record comparison
                image_id = line[2]
                if dataset_name == self.SUBSET_NAME_LIVE:
                    image_id = self.live2tid_ref_image(image_id)

                comparisons_per_image[image_id].append((jod, path_reference, path_distorted))

        for image_id in list(comparisons_per_image.keys()):
            if len(comparisons_per_image[image_id]) == 0:
                comparisons_per_image.pop(image_id)  # remove empty keys

        self.image_ids = sorted(list(comparisons_per_image.keys()))

        paths_ref, paths_dist, qs,  = [], [], []
        for i, image_id in enumerate(self.image_ids):
            comparisons = comparisons_per_image[image_id]
            # print(f'[index {i}] {image_id}')
            for jod, path_reference, path_distorted in comparisons:
                qs.append(jod)
                paths_ref.append(path_reference)
                paths_dist.append(path_distorted)

        dist_images_per_image = [len(comparisons_per_image[image_id]) for image_id in self.image_ids]
        self.process_dataset_data(qs, paths_ref, paths_dist, dist_images_per_image)

    def is_hdr_image(self, path):
        path = path.lower()
        return self.SUBSET_NAME_NARWARIA in path or self.SUBSET_NAME_KORSHUNOV in path


class UPIQHDRDataset(UPIQDataset):
    data_subsets = [
        UPIQDataset.SUBSET_NAME_KORSHUNOV,
        UPIQDataset.SUBSET_NAME_NARWARIA
    ]

    # Korshunov 20 + Narwaria 10
    num_ref_images = (20+10)  # 30 total

    def __init__(self, **kwargs):
        super().__init__(
            name_tag="HDR",
            is_hdr=True,

            # use_ref_img_cache=True,

            **kwargs
        )


class UPIQHDRFullDataset(UPIQDataset):
    data_subsets = [
        UPIQDataset.SUBSET_NAME_LIVE,
        UPIQDataset.SUBSET_NAME_TID,
        UPIQDataset.SUBSET_NAME_KORSHUNOV,
        UPIQDataset.SUBSET_NAME_NARWARIA
    ]

    # Korshunov 20 + Narwaria 10 + TID2013 25 + LIVE 10 (10 out of 29 are unique)
    num_ref_images = (20+10+25+10)  # 65 total

    def __init__(self, **kwargs):
        super().__init__(
            name_tag="HDR-Full",
            is_hdr=True,
            **kwargs
        )


class UPIQSDRDataset(UPIQDataset):
    data_subsets = [
        UPIQDataset.SUBSET_NAME_LIVE,
        UPIQDataset.SUBSET_NAME_TID,
    ]

    # TID2013 25 + LIVE 10 (10 out of 29 are unique)
    num_ref_images = (25+10)  # 30 total

    def __init__(self, **kwargs):
        super().__init__(
            name_tag="SDR",
            is_hdr=False,
            **kwargs
        )


class UPIQSDR2HDRDataset(UPIQSDRDataset):
    # SDR data converted to HDR: will use .exr not .png for LIVE and TID
    def __init__(self, **kwargs):
        super(UPIQSDRDataset,self).__init__(
            name_tag="SDR2HDR",
            is_hdr=True,
            **kwargs
        )


# test
if __name__ == "__main__":
    d = UPIQHDRDataset()
    d = UPIQHDRFullDataset()
    d = UPIQSDRDataset()
    d = UPIQSDR2HDRDataset()
