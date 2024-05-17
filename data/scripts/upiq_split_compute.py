import numpy as np
from mp_task.util import split_list

from data.datasets.upiq import UPIQHDRFullDataset
from training.train_config import SPLIT_NAME_TRAIN, SPLIT_NAME_VAL, SPLIT_NAME_TEST
from utils.logging import log_warn
from utils.misc.miscelaneous import MaintainRandomSeedConsistency


class UPIQ_DATA:
    TAG_LIVE = "l"
    TAG_TID2013 = "t"
    TAG_KORSHUNOV = "k"
    TAG_NARWARIA = "n"

    def __init__(self):
        log_warn("Setting up UPIQ Full dataset to compute cross-validation splits...")

        # get the dataset and the image order as they appear in the data (see upiq.py)
        self.upiq_dataset = UPIQHDRFullDataset()

        self.images = self.upiq_dataset.image_ids
        self.image_is_sdr = [image[0] in [self.TAG_TID2013, self.TAG_LIVE] for image in self.images]
        self.images_sdr = [image for (image, is_sdr) in zip(self.images, self.image_is_sdr) if is_sdr]
        self.images_hdr = [image for (image, is_sdr) in zip(self.images, self.image_is_sdr) if not is_sdr]


def upiq_get_split_test_dataset(
        split_sdr_datasets=True,  # will add 2 splits: 1. TID2013 and 2.LIVE
        split_hdr_datasets=True,  # will add 2 splits: 1. Korshunov and 2. Narwaria
        split_sdr=True,  # will add 1 split of SDR data (combined TID2013+LIVE)
        split_hdr=True,  # will add 1 split of HDR data (combined Korshunov+Narwaria)
        verbose=False
):
    upiq = UPIQ_DATA()

    nfold_splits_full, nfold_splits_sdr, nfold_splits_hdr = [], [], []

    def get_split_indices(images, test_tags):
        return (
            [i for i, image in enumerate(images) if image[0] not in test_tags],
            [],  # no validation
            [i for i, image in enumerate(images) if image[0] in test_tags]
        )

    def get_splits_for_test_set(test_tags):
        nfold_splits_full.append(get_split_indices(upiq.images, test_tags))
        nfold_splits_sdr.append(get_split_indices(upiq.images_sdr, test_tags))
        nfold_splits_hdr.append(get_split_indices(upiq.images_hdr, test_tags))

    # which dataset to hold out during training and use as test split
    split_data = []
    if split_sdr_datasets:
        split_data.append(("TID2013", [UPIQ_DATA.TAG_TID2013]))
        split_data.append(("LIVE", [UPIQ_DATA.TAG_LIVE]))
    if split_hdr_datasets:
        split_data.append(("NARWARIA", [UPIQ_DATA.TAG_NARWARIA]))
        split_data.append(("KORSHUNOV", [UPIQ_DATA.TAG_KORSHUNOV]))
    if split_sdr:
        split_data.append(("SDR", [UPIQ_DATA.TAG_LIVE, UPIQ_DATA.TAG_TID2013]))
    if split_hdr:
        split_data.append(("HDR", [UPIQ_DATA.TAG_KORSHUNOV, UPIQ_DATA.TAG_NARWARIA]))

    n_splits = len(split_data)

    for name, tags in split_data:
        get_splits_for_test_set(tags)

    split_names = [name for name, tags in split_data]

    if verbose:
        fold_tags = [name for name, tags in split_data]
        print_splits(n_splits, nfold_splits_full, upiq.images, "Full", fold_tags)
        print_splits(n_splits, nfold_splits_sdr, upiq.images_sdr, "SDR", fold_tags)
        print_splits(n_splits, nfold_splits_hdr, upiq.images_hdr, "HDR", fold_tags)

    return n_splits, nfold_splits_full, nfold_splits_sdr, nfold_splits_hdr, split_names


def print_splits(n_folds, nfolds_splits, names, tag, fold_tags=None):
    get_name = lambda names, indices: [names[index] for index in indices]
    print()
    for i in range(n_folds):
        fold_tag = "" if fold_tags is None else f" [{fold_tags[i] if isinstance(fold_tags, list) else fold_tags}]"
        print(f'{n_folds}-fold cross-validation ({tag}): fold #{i}{fold_tag}')
        print(SPLIT_NAME_TRAIN, nfolds_splits[i][0], get_name(names, nfolds_splits[i][0]))
        print(SPLIT_NAME_VAL, nfolds_splits[i][1], get_name(names, nfolds_splits[i][1]))
        print(SPLIT_NAME_TEST, nfolds_splits[i][2], get_name(names, nfolds_splits[i][2]))


def upiq_get_split_indices(n_folds=5, seed=None, verbose=False):
    upiq = UPIQ_DATA()

    with MaintainRandomSeedConsistency(seed):
        n_images = 65

        if len(upiq.images) != n_images:
            raise ValueError()

        n_train = 39
        n_val = 13
        n_test = 13

        n_sdr = 35
        n_hdr = 30

        if not (n_train + n_val + n_test == n_images and n_sdr + n_hdr == n_images):
            raise ValueError()

        def split_indices(n):
            p = np.random.permutation(n)
            splits = split_list(p, n_folds, append_leftover_to_last=False)
            split_images = np.zeros(n, int)
            for i, split in enumerate(splits):
                for j in split:
                    split_images[j] = i
            return split_images

        sdr_splits = split_indices(n_sdr)
        hdr_splits = split_indices(n_hdr)

        image_splits = list()
        i = j = 0
        for is_sdr in upiq.image_is_sdr:
            if is_sdr:
                image_splits.append(sdr_splits[i])
                i += 1
            else:
                image_splits.append(hdr_splits[j])
                j += 1

        split_counts_sdr = np.zeros(n_folds)
        split_counts_hdr = np.zeros(n_folds)
        for split_num in range(n_images):
            # print(upiq.images[split_num], upiq.image_is_sdr[split_num], image_splits[split_num])
            if upiq.image_is_sdr[split_num]:
                split_counts_sdr[image_splits[split_num]] += 1
            else:
                split_counts_hdr[image_splits[split_num]] += 1

        # v1
        nfold_splits_full = []
        for test_split in range(n_folds):
            val_split = (test_split + 1) % n_folds
            train, val, test = [], [], []
            for index, split_num in enumerate(image_splits):
                if split_num == test_split:
                    test.append(index)
                elif split_num == val_split:
                    val.append(index)
                else:
                    train.append(index)
            nfold_splits_full.append((train, val, test))

        def get_n_folds(split_nums):
            nfold_splits = []
            for test_split in range(n_folds):
                # given a test split index, use the next split for validation
                val_split = (test_split + 1) % n_folds
                train = [i for i, split_num in enumerate(split_nums) if
                         (split_num != test_split and split_num != val_split)]
                val = [i for i, split_num in enumerate(split_nums) if (split_num == val_split)]
                test = [i for i, split_num in enumerate(split_nums) if (split_num == test_split)]
                nfold_splits.append((train, val, test))
            return nfold_splits

        nfold_splits_sdr = get_n_folds(sdr_splits)
        nfold_splits_hdr = get_n_folds(hdr_splits)

        if verbose:
            print(upiq.upiq_dataset.image_ids)
            print_splits(n_folds, nfold_splits_full, upiq.images, "Full")
            print_splits(n_folds, nfold_splits_sdr, upiq.images_sdr, "SDR")
            print_splits(n_folds, nfold_splits_hdr, upiq.images_hdr, "HDR")

    return n_folds, nfold_splits_full, nfold_splits_sdr, nfold_splits_hdr


if __name__ == "__main__":
    upiq_get_split_test_dataset(verbose=True)
    # n_folds = 5
    # upiq_get_split_indices(n_folds=n_folds, seed=0, verbose=True)
