from collections import OrderedDict
from copy import deepcopy
import os, yaml
import shutil

from modules.VisionTransformer.backbone import VIT_VARIANT_B8, VIT_VARIANT_B16

from modules.vtamiq.vtamiq import VTAMIQ
from modules.PieAPP import PUPieAPP as PieAPP

from data.patch_datasets import dataset_split
from data.patch_sampling import GRID_TYPE_PERTURBED_SIMPLE

from utils.misc.correlations import *
from utils.logging import log_warn, log

# ************** CONSTANTS **************

DATASET_TID2013 = "TID2013"
DATASET_TID2008 = "TID2008"
DATASET_LIVE = "LIVE"
DATASET_CSIQ = "CSIQ"
DATASET_PIEAPP_TEST = "PieAPPTestset"
DATASET_PIEAPP_TRAIN = "PieAPPTrainset"
DATASET_KADID10K = "KADID10k"
DATASET_KONIQ10K = "KONIQ10k"
DATASET_UPIQ_SDR = "UPIQ-SDR"
DATASET_UPIQ_HDR = "UPIQ-HDR"
DATASET_UPIQ_SDR2HDR = "UPIQ-SDR2HDR"
DATASET_UPIQ_HDR_FULL = "UPIQ-HDR-Full"
DATASET_SIHDR = "SIHDR"

SPLIT_NAME_TRAIN = "Training"
SPLIT_NAME_VAL = "Validation"
SPLIT_NAME_TEST = "Testing"
SPLIT_NAME_FULL = "FullDataset"

SPLIT_TYPE_RANDOM = "random"
SPLIT_TYPE_INDICES = "indices"

PATCH_COUNT = "patch_count"
BATCH_SIZE = "batch_size"
SHUFFLE = "shuffle"
PATCH_FLIP = "allow_img_flip"
IMG_ZERO_ERORR_Q_PROB = "img_zero_error_q_prob"
USE_ALIGNED_PATCHES = "use_aligned_patches"
NUM_REPEATS_DATA = "num_repeats_data"
DATALOADER_PARAMS = "dataloader_params"
USE_DEFAULT_PARAMS = "use_default_params"

MODEL_VTAMIQ = "VTAMIQ"
MODEL_PIEAPP = "PIEAPP"

PU_WRAPPER_DISPLAY_NONE = "PU_WRAPPER_DISPLAY_NONE"
PU_WRAPPER_DISPLAY_RANDOMIZED = "PU_WRAPPER_DISPLAY_RANDOMIZED"

# NOTE: don't change these to support old model weights
MODEL_STATE_DICT = "model_state_dict"
PREF_MODULE_STATE_DICT = "pref_module_state_dict"

# ************** MODELS **************

models_vtamiq = {
    MODEL_VTAMIQ: VTAMIQ,
}

models_pieapp = {
    MODEL_PIEAPP: PieAPP
}

# ************** CONFIGS **************

global_config = OrderedDict(
    is_debug=False,
    is_verbose=True,

    is_domain_adaptation=False,

    dataloader_num_workers=-1,  # -1 to control this in script (based on dataset), otherwise to override
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    no_cuda=False,  # toggle to disable gpu

    do_train=False,
    do_val=False,
    do_test=True,

    # number of times to repeat computations for val/test splits
    num_repeats_val=1,
    num_repeats_test=4,

    train_save_latest=True,
    test_use_latest=True,  # use latest or best SROCC model for testing

    model=MODEL_VTAMIQ,

    use_pu=False,  # controls display simulation and PU encoding
    use_pref_module=False,  # additional module to remap Q

    dataset=DATASET_TID2013,
    dataset_test=None,
    allow_use_full_dataset=True,
    allow_use_full_dataset_test=True,

    load_checkpoint_file=None,

    # === TRAINING PARAMS ===

    seed=-1,  # -1 for random seed

    num_epochs=20,

    # === OPTIMIZER ===
    optimizer="AdamW",
    optimizer_weight_decay=0.01,  # 0.025

    optimizer_learning_rate=0.00005,
    # optimizer_learning_rate=0.0001,

    scheduler_step_per_batch=False,
    scheduler_type="lambda",  # ["multistep", "lambda", "cosine"]
    optimizer_learning_rate_decay_multistep=0.1,  # decay rate for multistep scheduler
    optimizer_learning_rate_decay_lambda_goal=0.01,  # end goal for LR value (goal = LR_end / LR_start)
    optimizer_learning_rate_decay_cosine=0.01,  # cosine scheduler ratio for LRmin = decay * LRmax
    optimizer_decay_after_n_epochs=[10, 15],
    optimizer_sgd_momentum=0.9,
    optimizer_sgd_nesterov=True,

    grad_scale=256,  # for torch.amp.autocast + gradient scaling

    # loss function weights
    # NOTE: MAE loss magnitude is usually ~10x larger than other losses, so assigning smaller weight may be appropriate.
    # training without MAE results in unpredictable scale of the output values (outside 0-1);
    # we don't use activation functions to ensure 0-1 (e.g. sigmoid)
    # weighted combination of the losses performs better.
    weight_mae_loss=1,
    weight_rank_loss=1,
    weight_pears_loss=1,

    # decays for loss function weights
    # example usage: want a schedule for losses, i.e. control the weight over time
    # use loss_decay < 1 to multiplicatively modify the weight of particular loss each epoch
    weight_mae_loss_decay=1.,
    weight_rank_loss_decay=1.,
    weight_pears_loss_decay=1.,

    # === LOGGING ===
    print_flops=False,
    print_params=False,

    checkpoint_every_n_batches=5000,  # useful for large datasets with many batches/epoch

    tensorlog_every_n_steps=10,  # fewer logs for tensorboard writer to limit the amount of logged data
    num_batches_for_correlation=8,  # writer will log current correlations for N latest batches

    output_tag="",
    output_dir="./output",
    output_txt="output.txt",
    debug_txt="debug.txt",

    save_optimizer=False,
    save_code_folder="source_code",

    save_val_outputs=True,
    save_test_outputs=True,
    save_test_outputs_txt="output_qs.txt",

    config_validated=False,
)

domain_adaptation_config = OrderedDict(
    loss_source_weight=1.0,
    loss_target_weight=1.0,
    loss_coral_weight=1.0,
    coral_loss_ema=1.0,  # 1.0 = only use current batch, otherwise apply EMA on cov s/t matrices
    coral_loss_min_len=5,  # smaller batch sizes may produce too much noise
    dataset_source=DATASET_KADID10K,  # which dataset to use as source
    num_repeats_source=1,
    num_repeats_target=1,
    source_use_full_split=False,
    max_num_workers_per_dataset=6,  # less than 8 to leave some cores for other work
    pu_wrapper_config_target=None,  # custom parameters for target pu wrapper
    pu_wrapper_config_source=None,  # custom parameters for source pu wrapper
    allow_single_dataloader=False,
    use_feats_separate=True,  # compute CORAL loss on ref/dist/diff separately
    use_feats_dist=True,  # include dist features for CORAL loss
    use_feats_diff=False,  # include diff features for CORAL loss
)


# ************** CONFIGURATION FOR MODELS **************

# default ViT backbone parameters
vit_config = OrderedDict(
    variant=VIT_VARIANT_B16,  # choose from [VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16]
    use_cls_token=True,
    pretrained=True,
    num_keep_layers=6,  # -1 for keeping all layers
    num_adapters=0,
    num_scales=1,  # num_scales<2 for no scale embedding
    num_extra_tokens=8,  # number of additional "register" tokens
    use_layer_scale=True,  # LayerScale from DeiT
    path_drop_prob=0.1,  # drop path probability from DeiT
)

vtamiq_config = OrderedDict(
    vit_config=vit_config,

    # channel-attention-based calibration network parameters (DiffNet)
    calibrate=True,  # apply calibration network to feature diff after ViT
    diff_scale=True,  # apply LayerScale to token difference vector
    num_rgs=4,  # RGs in difference modulation
    num_rcabs=4,  # RCABs per RG
    ca_reduction=16,  # channel downsample ratio for CA
    rg_path_drop=0.1,  # for RGs residual connections

    # quality/weight predictor MLPs
    predictor_dropout=0.1,
)

pieapp_config = OrderedDict(
    pretrained=True,  # allow pretrained PieAPP (original or VGG16)
    pretrained_vgg=True,  # use VGG16 weights instead of PieAPP pth
)


def get_model_type_and_config(model_name):
    # VTAMIQ
    if model_name in models_vtamiq:
        model_type = models_vtamiq[model_name]
        model_config = vtamiq_config

    # PieAPP
    elif model_name in models_pieapp:
        model_type = models_pieapp[model_name]
        model_config = {}

    else:
        raise ValueError(f"Unsupported model {model_name}")

    return model_type, model_config


pretraining_config = OrderedDict(
    # toggle use of pretrained ViT (Imagenet)
    allow_pretrained_vit=True,

    # if using pretrained model (not just pretrained ViT), toggle loading ViT and Diffnet weights
    allow_pretrained_weights=True,
    allow_pretrained_weights_vit=True,
    allow_pretrained_weights_diffnet=True,
)

freeze_config = OrderedDict(
    # Note: freezing features is useful when quality predictor is untrained, but pretrained ViT is used.
    # We don't want to overwrite pretrained BERT features while quality predictor is outputting garbage.
    # Instead, freeze the transformer, spend several epochs training quality predictor, then unfreeze the transformer
    # for combined fine-tuning.
    freeze_vtamiq=False,  # global toggle to allow freezing VTAMIQ
    freeze_pieapp=False,  # global toggle to allow freezing PieAPP

    freeze_conditional=False,  # allow freezing based on dataset and checkpoint parameters

    # when to end freezing ViT weights (based on dataset); will unfreeze when freeze_end_after_epochs < epoch
    freeze_end_after_epochs={
        DATASET_TID2013: 2,
        DATASET_TID2008: 2,
        DATASET_LIVE: 3,
        DATASET_CSIQ: 2,
        DATASET_PIEAPP_TRAIN: 1,
        DATASET_PIEAPP_TEST: 0,
        DATASET_KADID10K: 1,
        DATASET_KONIQ10K: 1,
        DATASET_UPIQ_SDR: 1,
        DATASET_UPIQ_HDR: 1,
        DATASET_UPIQ_HDR_FULL: 1,
        DATASET_UPIQ_SDR2HDR: 1,
        DATASET_SIHDR: 0,
    }
)

freeze_dict_vit = OrderedDict(
    freeze_encoder=True,  # ViT encoder
    freeze_encoder_adapters=False,  # ViT encoder layer Adapter modules
    freeze_encoder_layerscale=False,  # ViT encoder layer LayerScale modules
    freeze_embeddings_patch=True,  # ViT patch embeddings
    freeze_embeddings_cls_token=True,  # ViT embedding tokens
    freeze_embeddings_extra_tokens=True,  # ViT embedding tokens
    freeze_embeddings_pos=True,  # ViT positional embeddings
    freeze_embeddings_scale=False,  # ViT scale embeddings
)

freeze_dict_vtamiq = OrderedDict(
    freeze_dict_vit=freeze_dict_vit,
    freeze_quality_decoder=False,
    freeze_q_predictor=False,
    freeze_w_predictor=False,
)

freeze_dict_pieapp = OrderedDict(
    freeze_feature_extractor=True,
    freeze_comparitor=False,
)

pref_module_config = OrderedDict(
    weight=6.,  # q' = w * q + b; w=6 approximately maps sigmoid output into preferences using the JOD unit model
)

# PU transform parameters
pu_wrapper_config_base = OrderedDict(
    normalize_mean_std=True,  # normalize mean/std (subtract, divide)
    normalize_mean_std_imagenet=False,  # 0.5 False or imagenet mean/std True
    normalize_pu=True,  # toggle to normalize from PU-encoded units to range 0-1 or 0-2.3
    normalize_pu_range_srgb=False,  # toggle to use 0-2.3 normalization (else 0-1)
)

# base config, custom parameters from wrapper type configs will overwrite these values as relevant
pu_wrapper_config_display_base = OrderedDict(
    display_L_max=100,  # highest display luminance for white color
    display_L_min=2.5,  # lowest display luminance for white color (not L_blk)
    display_L_cr=1000,  # contrast ratio -> L_blk=L_max/L_cr=100/1000=0.1cd/m2
    display_reflectivity=0.005,  # NOTE: lowered from default 0.01 (1%)
)

pu_wrapper_config_display_ambient_random = OrderedDict(
    # randomization controls
    randomize_ambient=True,
    randomize_L_max=True,
    # randomization params
    rand_distrib_normal=True,  # use uniform or normal distribution
    rand_distrib_ambient_mean=100,  # ambient lux mean
    rand_distrib_ambient_delta=25,  # ambient lux deviation (sigma if using normal distribution)
    rand_L_max_delta=10,  # maximum random deviation from L_max (1 sigma deviation if rand_distrib_normal==True)
)

pu_wrapper_config_displays = {
    PU_WRAPPER_DISPLAY_RANDOMIZED: pu_wrapper_config_display_ambient_random,
}


# ************** DATASETS **************

# this will be passed to each dataset via __init__
dataset_config_base = OrderedDict(
    full_reference=True,  # False to force NR IQA; only used for FR datasets
    # NOTE: full_reference=False is deprecated

    # resolution="half",  # only relevant for KONIQ10k dataset (it has full and half res), otherwise ignored

    patch_dim=-1,  # this depends on ViT configuration and will be updated by validate_configs()
    patch_num_scales=1,  # 5 scales -> when patch_dim=16: {0: 16, 1: 32, 2: 64, 3: 128, 4: 256}

    normalize=True,  # apply normalization on preprocess
    normalize_imagenet=False,  # normalize using imagenet's mean and std dev

    patch_sampling_num_scales_ratio=1.75,  # larger value leads to fewer samples coming from large scales

    patch_sampler_config=OrderedDict(
        # centerbias_weight=0.25,
        # diff_weight=1,
        uniform_weight=0.1,
        grid_type=GRID_TYPE_PERTURBED_SIMPLE,
        # GRID_TYPE_PERTURBED_SIMPLE is grid-based sampling with random perturbation
    ),
)


def dataset_target():
    return global_config["dataset"]


def dataset_source_da():
    return domain_adaptation_config["dataset_source"]


dataset_split_config_base = OrderedDict(
    split_type=SPLIT_TYPE_INDICES,  # pick from [SPLIT_TYPE_INDICES, SPLIT_TYPE_RANDOM]
)

num_workers_config = {
    DATASET_UPIQ_HDR: 6,
    DATASET_UPIQ_HDR_FULL: 8,
    DATASET_UPIQ_SDR2HDR: 8,
    DATASET_SIHDR: 6,

    DATASET_LIVE: 6,
    DATASET_TID2008: 6,
    DATASET_CSIQ: 6,

    DATASET_TID2013: 8,
    DATASET_PIEAPP_TEST: 4,

    DATASET_KONIQ10K: 8,
    DATASET_UPIQ_SDR: 8,
    DATASET_KADID10K: 8,
    DATASET_PIEAPP_TRAIN: 8,
}

# ************** DATASET SPLIT PARAMS **************

# default parameters
dataloader_config_base = {
    SPLIT_NAME_TRAIN: {
        BATCH_SIZE: 16,  # x256: [train/test] 20, 12 (pairwise); x512: 8, 5 (pairwise)
        SHUFFLE: True,
        PATCH_COUNT: 384,
        PATCH_FLIP: True,
        IMG_ZERO_ERORR_Q_PROB: -1,
        USE_ALIGNED_PATCHES: True,  # aligned over Ref and Distorted images
        NUM_REPEATS_DATA: 1,
    },
    SPLIT_NAME_VAL: {
        BATCH_SIZE: 16,
        SHUFFLE: False,
        PATCH_COUNT: 1024,
        PATCH_FLIP: False,
        IMG_ZERO_ERORR_Q_PROB: -1,
        USE_ALIGNED_PATCHES: True,
        NUM_REPEATS_DATA: 1,
    },
    SPLIT_NAME_TEST: {
        BATCH_SIZE: 16,
        SHUFFLE: False,
        PATCH_COUNT: 1024,
        PATCH_FLIP: False,
        IMG_ZERO_ERORR_Q_PROB: -1,
        USE_ALIGNED_PATCHES: True,
        NUM_REPEATS_DATA: 1,
    },
    DATALOADER_PARAMS: {
        # when USE_DEFAULT_PARAMS=true, will use parameters as listed above
        # when USE_DEFAULT_PARAMS=false, validate_configs() will reconfigure some parameters given the current run
        USE_DEFAULT_PARAMS: False
    }
}

# custom dataloader params for vtamiq
dataloader_config_vtamiq = {
    SPLIT_NAME_TRAIN: {
        BATCH_SIZE: 16,
        PATCH_COUNT: 384,
    },
    SPLIT_NAME_VAL: {
        BATCH_SIZE: 16,
        PATCH_COUNT: 512,
    },
    SPLIT_NAME_TEST: {
        BATCH_SIZE: 16,
        PATCH_COUNT: 512,
    },
}

# custom dataloader params for pieapp
dataloader_config_pieapp = {
    SPLIT_NAME_TRAIN: {
        BATCH_SIZE: 8,
        PATCH_COUNT: 96,
    },
    SPLIT_NAME_VAL: {
        BATCH_SIZE: 3,
        PATCH_COUNT: 512,
    },
    SPLIT_NAME_TEST: {
        BATCH_SIZE: 3,
        PATCH_COUNT: 1024,
    },
}


def setup_split_indices(ind):
    if isinstance(ind, list):
        return ind
    elif isinstance(ind, tuple):
        if 3 < len(ind):
            raise ValueError(f"Unsupported tuple format for range-based split indices: [{ind}]")
        return [i for i in range(*ind)]
    elif isinstance(ind, int):
        if ind == 0:
            return [0]
        return [i for i in range(ind)]
    else:
        raise ValueError(f"Unsupported index format for split indices: [{ind}]")


def setup_split_config(i_n1, i_n2, i_n3):
    return {
        SPLIT_NAME_TRAIN: setup_split_indices(i_n1),
        SPLIT_NAME_VAL: setup_split_indices(i_n2),
        SPLIT_NAME_TEST: setup_split_indices(i_n3),
    }


# use 6-2-2 split ratio when splitting dataset randomly
split_config_random = {
    SPLIT_NAME_TRAIN: 6,
    SPLIT_NAME_VAL: 2,
    SPLIT_NAME_TEST: 2,
}

split_config_tid2013 = setup_split_config(15, (15, 20), (20, 25))

# TID 2008 has the same format as TID2013, but with less distorted images. Hence the same config can be used
split_config_tid2008 = deepcopy(split_config_tid2013)

split_config_live = setup_split_config(17, (17, 23), (23, 28))

split_config_csiq = setup_split_config(18, (18, 24), (24, 30))

split_config_pieapptrain = setup_split_config(130, (130, 135), (135, 140))  # used for training

split_config_pieapptest = setup_split_config(0, 0, 40)  # used for testing

split_config_kadid10k = setup_split_config(49, (49, 65), (65, 81))

split_config_upiq_sdr = setup_split_config(21, (21, 28), (28, 35))  # LIVE 10 + TID 25
split_config_upiq_hdr = setup_split_config(18, (18, 24), (24, 30))  # korshunov 20 + narwaria 10
split_config_upiq_full = setup_split_config(39, (39, 52), (52, 65))  # k + n + l + t

split_config_sihdr = setup_split_config(181, 0, 0)  # all 181 HDR images in the dataset

split_config_koniq10k = setup_split_config(6045, (6045, 8059), (8059, 10073))


def get_dataset_configs(dataset_name):
    validate_configs_check()

    # CLASSICAL IQA
    if dataset_name == DATASET_TID2013:
        from data.datasets.tid import TID2013Dataset
        dataset_type = TID2013Dataset
        splits_config = split_config_tid2013

    elif dataset_name == DATASET_TID2008:
        from data.datasets.tid import TID2008Dataset
        dataset_type = TID2008Dataset
        splits_config = split_config_tid2008

    elif dataset_name == DATASET_LIVE:
        from data.datasets.live import LIVEDataset
        dataset_type = LIVEDataset
        splits_config = split_config_live

    elif dataset_name == DATASET_CSIQ:
        from data.datasets.csiq import CSIQDataset
        dataset_type = CSIQDataset
        splits_config = split_config_csiq

    # KADID
    elif dataset_name == DATASET_KADID10K:
        from data.datasets.kadid10k import KADID10kDataset
        dataset_type = KADID10kDataset
        splits_config = split_config_kadid10k

    elif dataset_name == DATASET_UPIQ_SDR:
        from data.datasets.upiq import UPIQSDRDataset
        dataset_type = UPIQSDRDataset
        splits_config = split_config_upiq_sdr

    elif dataset_name == DATASET_UPIQ_HDR:
        from data.datasets.upiq import UPIQHDRDataset
        dataset_type = UPIQHDRDataset
        splits_config = split_config_upiq_hdr

    elif dataset_name == DATASET_UPIQ_HDR_FULL:
        from data.datasets.upiq import UPIQHDRFullDataset
        dataset_type = UPIQHDRFullDataset
        splits_config = split_config_upiq_full

    elif dataset_name == DATASET_UPIQ_SDR2HDR:
        from data.datasets.upiq import UPIQSDR2HDRDataset
        dataset_type = UPIQSDR2HDRDataset
        splits_config = split_config_upiq_sdr

    elif dataset_name == DATASET_SIHDR:
        from data.datasets.sihdr import SIHDRDataset
        dataset_type = SIHDRDataset
        splits_config = split_config_sihdr

    # PieAPP
    elif dataset_name == DATASET_PIEAPP_TRAIN:
        from data.datasets.pieapp_dataset import PieAPPTrainPairwise
        dataset_type = PieAPPTrainPairwise
        splits_config = split_config_pieapptrain

    elif dataset_name == DATASET_PIEAPP_TEST:
        from data.datasets.pieapp_dataset import PieAPPTestset
        dataset_type = PieAPPTestset
        splits_config = split_config_pieapptest

    # NO-REFERENCE DATASETS
    elif dataset_name == DATASET_KONIQ10K:
        from data.datasets.koniq10k import KONIQ10k
        dataset_type = KONIQ10k
        splits_config = split_config_koniq10k

    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(dataset_name))

    splits_indices = get_dataset_splits(dataset_type, splits_config)

    return dataset_type, splits_indices


def get_dataset_splits(dataset, dataset_split_config):
    split_type = dataset_split_config_base["split_type"]

    if split_type == SPLIT_TYPE_INDICES:
        print("Using predefined split indices.")

        return dataset_split_config

    elif split_type == SPLIT_TYPE_RANDOM:
        print("Using random split indices.")

        # compute number of images per split given the number of reference images in the dataset and the split ratios
        num_ref_images = dataset.num_ref_images  # number of reference images in the dataset
        num_total = sum(split_config_random.values())  # sum of split ratios (ex: 6-2-2 -> 6+2+2=10)
        split_counts = {}
        for split_name in split_config_random:
            # compute the number of images in current split given the required ratio of all reference images
            split_counts[split_name] = int(split_config_random[split_name] / num_total * num_ref_images)
        num_total = sum(split_counts.values())  # sum of all images in the splits
        # if there is leftover, add it to the Train set
        if num_ref_images != num_total:
            split_counts[SPLIT_NAME_TRAIN] = split_counts[SPLIT_NAME_TRAIN] + num_ref_images - num_total

        # random permutation of the available reference image indices
        split_indices = np.random.permutation(num_ref_images)

        splits = {}
        total = 0
        for split_name in split_counts:
            split_total = split_counts[split_name]  # number of indices in current split
            if split_total < 1:
                log_warn(f"get_dataset_splits(), split {split_name} got zero images.")
                continue
            splits[split_name] = sorted(deepcopy(split_indices[total: total + split_total]))  # slice
            total += split_total

        return splits

    else:
        raise ValueError("TrainConfig: unsupported split_type {}.".format(split_type))


def make_dataset_with_config_splits(dataset_name):
    validate_configs_check()

    dataset_config = deepcopy(dataset_config_base)

    dataset_type, split_config = get_dataset_configs(dataset_name)

    dataset = dataset_type(
        **dataset_config
    )

    # add full dataset split
    split = dataset_split(name=SPLIT_NAME_FULL, indices=None)
    dataset.add_split(split)

    # add train/val/test splits
    for split_name in split_config:
        split = dataset_split(name=split_name, indices=split_config[split_name])
        if 0 < len(split.indices):
            dataset.add_split(split)

    return dataset


class DatasetFactory(object):
    def __init__(self,
                 ):
        self.dataset_cache = {}

    def get_dataset(self, dataset_name):
        if dataset_name in self.dataset_cache:
            dataset = self.dataset_cache[dataset_name]
        else:
            dataset = make_dataset_with_config_splits(dataset_name)
            self.dataset_cache[dataset_name] = dataset

        # if no split is selected, select the first split available in split dict
        if dataset.split_name_crt is None:
            if 0 < len(dataset.splits_dict.keys()):
                dataset.set_split_crt(sorted(dataset.splits_dict.keys())[0])
            else:
                log_warn(f"Dataset {dataset.name} has no splits...")

        return dataset

    def get_dataloader(self, dataset_name, split_name, dataloader_params):
        dataset = self.get_dataset(dataset_name)

        if not dataset.has_split(split_name):
            log_warn(f"Dataset {dataset.name} does not contain split (split_name={split_name}). Dataset will be None.")
            return None

        from data.patch_datasets import PatchDatasetLoader
        split_loader = PatchDatasetLoader(
            dataset=dataset,
            split_name=split_name,
            batch_size=dataloader_params[BATCH_SIZE],
            patch_count=dataloader_params[PATCH_COUNT],
            allow_img_flip=dataloader_params[PATCH_FLIP],
            img_zero_error_q_prob=dataloader_params[IMG_ZERO_ERORR_Q_PROB],
            use_aligned_patches=dataloader_params[USE_ALIGNED_PATCHES],
            shuffle=dataloader_params[SHUFFLE],
            num_repeats_data=dataloader_params[NUM_REPEATS_DATA],
            num_workers=global_config["dataloader_num_workers"],
            pin_memory=global_config["dataloader_pin_memory"],
            persistent_workers=global_config["dataloader_persistent_workers"],
        )
        return split_loader


def get_dataloaders(use_full_dataset=False, dataloader_config=None):
    validate_configs_check()

    if dataloader_config is None:
        dataloader_config = deepcopy(dataloader_config_base)

    if use_full_dataset:
        log_warn("use_full_dataset=True; all dataloaders will use FULL dataset (all reference images).")

    dataset_factory = DatasetFactory()

    dataset_name = dataset_target()
    get_split_name = lambda split_name, use_full_dataset: SPLIT_NAME_FULL if use_full_dataset else split_name

    loader_train = dataset_factory.get_dataloader(
        dataset_name, get_split_name(SPLIT_NAME_TRAIN, use_full_dataset), dataloader_config[SPLIT_NAME_TRAIN])

    loader_val = dataset_factory.get_dataloader(
        dataset_name, get_split_name(SPLIT_NAME_VAL, use_full_dataset), dataloader_config[SPLIT_NAME_VAL])

    # check if test dataset is different from train dataset
    if global_config["dataset_test"] is not None:
        dataset_name = global_config["dataset_test"]

    use_full_dataset = use_full_dataset and global_config["allow_use_full_dataset_test"]

    loader_test = dataset_factory.get_dataloader(
        dataset_name, get_split_name(SPLIT_NAME_TEST, use_full_dataset), dataloader_config[SPLIT_NAME_TEST])

    return loader_train, loader_val, loader_test, dataset_factory


# ************** FUNCTIONS **************

def model_uses_scales():
    model_name = global_config["model"]
    if model_name in models_vtamiq:
        return 1 < vit_config["num_scales"]
    return False


def dataset_uses_scales():
    return 1 < dataset_config_base["patch_num_scales"]


def training_run_uses_scales():
    return dataset_uses_scales() and model_uses_scales()


def dataset_is_pairwise(dataset_name):
    return dataset_name == DATASET_PIEAPP_TRAIN


def dump_config_file(output_dir, config, name):
    path = os.path.join(output_dir, "{}.yaml".format(name))
    with open(path, "w") as f:
        yaml.dump(dict(config), f, sort_keys=False)


def save_configs(output_dir):
    validate_configs_check()

    dump_config_file(output_dir, global_config, "config")

    model_type, model_config = get_model_type_and_config(global_config["model"])
    dump_config_file(output_dir, model_config, "model_config")

    dump_config_file(output_dir, pretraining_config, "pretraining_config")

    if freeze_config["freeze_vtamiq"] and global_config["model"] in models_vtamiq:
        dump_config_file(output_dir, freeze_dict_vtamiq, "freeze_dict_vtamiq")
        dump_config_file(output_dir, freeze_config, "freeze_config")

    if freeze_config["freeze_pieapp"] and global_config["model"] == MODEL_PIEAPP:
        dump_config_file(output_dir, freeze_dict_pieapp, "freeze_dict_pieapp")
        dump_config_file(output_dir, freeze_config, "freeze_config")

    dump_config_file(output_dir, dataset_config_base, "dataset_config_base")
    dump_config_file(output_dir, dataloader_config_base, "dataloader_config_base")

    if global_config["use_pu"]:
        dump_config_file(output_dir, pu_wrapper_config_base, "pu_wrapper_config_base")
        dump_config_file(output_dir, pu_wrapper_config_display_base, "pu_wrapper_config_display_base")
        dump_config_file(output_dir, pu_wrapper_config_display_ambient_random, "pu_wrapper_config_display_random")

    if global_config["use_pref_module"]:
        dump_config_file(output_dir, pref_module_config, "pref_module_config")

    dataset_is_used_ = lambda database_name: (dataset_target() == database_name or
                                              (global_config["dataset_test"] is not None and
                                               global_config["dataset_test"] == database_name))
    dataset_is_used = lambda database_name: dataset_is_used_(database_name)

    if global_config["is_domain_adaptation"]:
        dump_config_file(output_dir, domain_adaptation_config, "domain_adaptation_config")
        dataset_is_used = lambda database_name: dataset_is_used_(database_name) or dataset_source_da() == database_name

    if dataset_is_used(DATASET_TID2013):
        dump_config_file(output_dir, split_config_tid2013, "tid2013_split_config")
    elif dataset_is_used(DATASET_TID2008):
        dump_config_file(output_dir, split_config_tid2008, "tid2008_split_config")
    elif dataset_is_used(DATASET_LIVE):
        dump_config_file(output_dir, split_config_live, "live_split_config")
    elif dataset_is_used(DATASET_KADID10K):
        dump_config_file(output_dir, split_config_kadid10k, "kadid10k_split_config")
    elif dataset_is_used(DATASET_UPIQ_SDR):
        dump_config_file(output_dir, split_config_upiq_sdr, "upiq_sdr_split_config")
    elif dataset_is_used(DATASET_UPIQ_HDR):
        dump_config_file(output_dir, split_config_upiq_hdr, "upiq_hdr_split_config")
    elif dataset_is_used(DATASET_UPIQ_HDR_FULL):
        dump_config_file(output_dir, split_config_upiq_full, "upiq_full_split_config")
    elif dataset_is_used(DATASET_SIHDR):
        dump_config_file(output_dir, split_config_sihdr, "split_config_sihdr")
    elif dataset_is_used(DATASET_KONIQ10K):
        dump_config_file(output_dir, split_config_koniq10k, "koniq10k_split_config")
    elif dataset_is_used(DATASET_CSIQ):
        dump_config_file(output_dir, split_config_csiq, "csiq_split_config")
    elif dataset_is_used(DATASET_PIEAPP_TRAIN):
        dump_config_file(output_dir, split_config_pieapptrain, "pieapptrain_split_config")
    elif dataset_is_used(DATASET_PIEAPP_TEST):
        dump_config_file(output_dir, split_config_pieapptest, "pieapp_split_config")


def save_code(output_dir):
    validate_configs_check()

    output_folder = global_config["save_code_folder"]
    dir = os.path.join(output_dir, output_folder)
    os.makedirs(dir)

    shutil.copyfile("./data/patch_sampling.py", os.path.join(dir, "patch_sampling.py"))
    shutil.copyfile("./data/patch_datasets.py", os.path.join(dir, "patch_datasets.py"))

    # save model code
    if "vtamiq" in global_config["model"].lower():
        shutil.copyfile("./modules/RCAN/channel_attention.py", os.path.join(dir, "channel_attention.py"))
        shutil.copyfile("./modules/VisionTransformer/transformer.py", os.path.join(dir, "transformer.py"))
        shutil.copyfile("./modules/VisionTransformer/backbone.py", os.path.join(dir, "backbone.py"))

        if global_config["model"] in models_vtamiq:
            shutil.copyfile("./modules/vtamiq/vtamiq.py", os.path.join(dir, "vtamiq.py"))

    elif global_config["model"] == MODEL_PIEAPP:
        shutil.copyfile("./modules/PieAPP/pu_pieapp.py", os.path.join(dir, "pieapp.py"))

    else:
        log_warn("Not saving model code.")

    if global_config["use_pu"]:
        shutil.copyfile("./modules/display_simulation/pu_display_wrapper.py",
                        os.path.join(dir, "pu_display_wrapper.py"))
        shutil.copyfile("./modules/display_simulation/display_model.py", os.path.join(dir, "display_model.py"))

    shutil.copyfile("./training/train.py", os.path.join(dir, "train.py"))


def validate_configs_check():
    if not global_config["config_validated"]:
        raise RuntimeError("Configs must be validated.")


def validate_configs():
    log("*** Validating config files...")

    # configure patch size
    if global_config["model"] in models_vtamiq and VIT_VARIANT_B8 == vtamiq_config["vit_config"]["variant"]:
        dataset_config_base["patch_dim"] = 8

    elif "pieapp" in global_config["model"].lower():
        dataset_config_base["patch_dim"] = 64
        dataset_config_base["patch_num_scales"] = 1
        log_warn("Model PieAPP uses patches with 1 scale and 64x64 size.")

        # need to normalize data with ImageNet mean/std not 0.5, 0.5, 0.5
        if pieapp_config["pretrained"] and pieapp_config["pretrained_vgg"]:
            pu_wrapper_config_base["normalize_mean_std_imagenet"] = True
            dataset_config_base["normalize_imagenet"] = True
            log_warn("Model PieAPP with VGG16 weights uses Imagenet mean/std for normalization.")

    else:
        dataset_config_base["patch_dim"] = 16

    vit_config["num_scales"] = max(1, vit_config["num_scales"])
    dataset_config_base["patch_num_scales"] = max(1, dataset_config_base["patch_num_scales"])

    if model_uses_scales() != dataset_uses_scales():
        raise ValueError(f"Mismatch between model/dataset use of different scales:"
                         f"Model {global_config['model']} {'uses' if model_uses_scales() else 'does not use'} scales "
                         f"while dataset {'uses' if dataset_uses_scales() else 'does not use'} scales.")

    log(f"Set dataset_config_base['patch_dim']={dataset_config_base['patch_dim']}.")

    if dataloader_config_base[DATALOADER_PARAMS][USE_DEFAULT_PARAMS]:
        log_warn("validate_configs() will not set batch size and patch count. Using defaults...")

    else:
        model_name = global_config["model"]
        if model_name in models_vtamiq:
            log("Configuring batch size and patch count for VTAMIQ.")
            data_config = dataloader_config_vtamiq
        elif "pieapp" in global_config["model"].lower():
            log("Configuring batch size and patch count for PieAPP.")
            data_config = dataloader_config_pieapp
        else:
            data_config = dataloader_config_base  # default

        # also apply domain adaptation modifier to HALF the batch size when DA is used
        # this is needed because with DA, we essentially process 2 batches at the same time
        da_mod = 2 if global_config["is_domain_adaptation"] else 1

        dataloader_config_base[SPLIT_NAME_TRAIN][BATCH_SIZE] = data_config[SPLIT_NAME_TRAIN][BATCH_SIZE] // da_mod
        dataloader_config_base[SPLIT_NAME_TRAIN][PATCH_COUNT] = data_config[SPLIT_NAME_TRAIN][PATCH_COUNT]

        dataloader_config_base[SPLIT_NAME_VAL][BATCH_SIZE] = data_config[SPLIT_NAME_VAL][BATCH_SIZE]
        dataloader_config_base[SPLIT_NAME_VAL][PATCH_COUNT] = data_config[SPLIT_NAME_VAL][PATCH_COUNT]

        dataloader_config_base[SPLIT_NAME_TEST][BATCH_SIZE] = data_config[SPLIT_NAME_TEST][BATCH_SIZE]
        dataloader_config_base[SPLIT_NAME_TEST][PATCH_COUNT] = data_config[SPLIT_NAME_TEST][PATCH_COUNT]

    log(f"Using train/val/test batch_size=["
        f"{dataloader_config_base[SPLIT_NAME_TRAIN][BATCH_SIZE]}, "
        f"{dataloader_config_base[SPLIT_NAME_VAL][BATCH_SIZE]}, "
        f"{dataloader_config_base[SPLIT_NAME_TEST][BATCH_SIZE]}].")

    log(f"Using train/val/test patch_count=["
        f"{dataloader_config_base[SPLIT_NAME_TRAIN][PATCH_COUNT]}, "
        f"{dataloader_config_base[SPLIT_NAME_VAL][PATCH_COUNT]}, "
        f"{dataloader_config_base[SPLIT_NAME_TEST][PATCH_COUNT]}].")

    if global_config["dataloader_num_workers"] == -1:
        num_workers = num_workers_config[dataset_target()]

        if global_config["is_domain_adaptation"]:
            # also check dataset source if using domain adaptation
            num_workers = max(num_workers, num_workers_config[dataset_source_da()])  # use the larger num workers

            # Note: hyperthreading doesn't help when all cores are already busy
            num_workers = min(domain_adaptation_config["max_num_workers_per_dataset"], num_workers)

        global_config["dataloader_num_workers"] = num_workers

        log(f"Set global_config['dataloader_num_workers']={global_config['dataloader_num_workers']}.")

    if dataset_target() == DATASET_PIEAPP_TRAIN:
        log_warn("training with PieAPP train dataset; Pairwise training mode will be used.")

    if global_config["use_pu"]:
        log_warn("Using display model and PU encoding (dataset normalization disabled)")
        dataset_config_base["normalize"] = False
        dataset_config_base["normalize_imagenet"] = False

    log("*** Config files successfully validated.")

    # update validation flag
    global_config["config_validated"] = True

# print("WARNING: call validate_configs() if externally modifying config dicts.")
