import time

from data.scripts.upiq_split_compute import upiq_get_split_indices, upiq_get_split_test_dataset
from training.train_config import *
from training.train_multi import parse_runs
from training import train
from utils.logging import FileLogger


########################################################################################################################
# Domain adaptation variants
DA_TYPE_S_HU_SIHDR = "S_HU_SIHDR"  # unlabeled HDR (SI-HDR)
DA_TYPE_S_HU_KADID = "S_HU_KADID"  # unlabeled SDR (KADID)
DA_TYPE_S_HU_UPIQS = "S_HU_UPIQS"  # unlabeled SDR (UPIQ)
DA_TYPE_S_HS = "S_HS"  # labeled SDR (KADID), synthetic labeled HDR (HDR KADID)
DA_TYPE_S_HL = "S_HL"  # labeled SDR (SDR UPIQ), labeled HDR (HDR UPIQ)


########################################################################################################################


def upiq_common_settings():
    """
    configures common parameters for a training run on UPIQ
    """
    pu_wrapper_config_display_ambient_random["randomize_ambient"] = False
    pu_wrapper_config_display_ambient_random["randomize_L_max"] = False

    global_config["dataset"] = DATASET_UPIQ_HDR_FULL

    global_config["use_pu"] = True

    # we specifically want to test based on INDICES and never the full set unless specified by INDICES
    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES
    global_config["allow_use_full_dataset"] = False

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = True

    global_config["train_save_latest"] = True

    global_config["num_epochs"] = 15

    global_config["optimizer_learning_rate"] = 0.0001
    global_config["scheduler_type"] = "lambda"
    global_config["optimizer_learning_rate_decay_multistep"] = 0.2

    global_config["scheduler_step_per_batch"] = False


def set_split_fold(splits, splits_sdr, splits_hdr, fold_num):
    split_config_upiq_full[SPLIT_NAME_TRAIN] = splits[fold_num][0]
    split_config_upiq_full[SPLIT_NAME_VAL] = splits[fold_num][1]
    split_config_upiq_full[SPLIT_NAME_TEST] = splits[fold_num][2]

    split_config_upiq_sdr[SPLIT_NAME_TRAIN] = splits_sdr[fold_num][0]
    split_config_upiq_sdr[SPLIT_NAME_VAL] = splits_sdr[fold_num][1]
    split_config_upiq_sdr[SPLIT_NAME_TEST] = splits_sdr[fold_num][2]

    split_config_upiq_hdr[SPLIT_NAME_TRAIN] = splits_hdr[fold_num][0]
    split_config_upiq_hdr[SPLIT_NAME_VAL] = splits_hdr[fold_num][1]
    split_config_upiq_hdr[SPLIT_NAME_TEST] = splits_hdr[fold_num][2]


def set_split_run_datasets_da(use_domain_adaptation):
    """
    this function configures data parameters for training with/without domain adaptation
    """
    if use_domain_adaptation:
        global_config["num_epochs"] = 30

        global_config["dataset"] = DATASET_UPIQ_HDR  # target
        domain_adaptation_config["dataset_source"] = DATASET_UPIQ_SDR2HDR  # source

        # Note: there is roughly 10 times more SDR data than HDR data
        # we can optionally repeat HDR subset N times in one epoch (for approx balance)
        domain_adaptation_config['num_repeats_target'] = 1
        domain_adaptation_config['num_repeats_source'] = 1  # repeat SDR subset once in one epoch

        # select dataset for testing
        hdr_only = True  # note: test on hdr only
        if hdr_only:
            global_config["dataset_test"] = DATASET_UPIQ_HDR  # test only on the hdr subset
        else:
            global_config["dataset_test"] = DATASET_UPIQ_HDR_FULL  # test on a mix of sdr/hdr data
    else:
        global_config["num_epochs"] = 20

        global_config["dataset"] = DATASET_UPIQ_HDR_FULL
        global_config["dataset_test"] = None


########################################################################################################################


def run_n_splits(use_domain_adaptation, n_splits, splits_full, splits_sdr, splits_hdr, split_names, run_tag):
    """
    this function runs N training/testing sessions (given N splits)
    IMPORTANT: run_splits() only works correctly for 1 model type per run
    otherwise, multiple settings get incorrectly mixed and bad things happen
    to properly isolate this behaviour, probably need to rewrite the whole training setup to use command line style
    instead of dictionaries for configuration...
    :param use_domain_adaptation: toggle to use DA during training
    :param n_splits: number of split
    :param splits_full: N lists (N splits, 1 list for each split), indices of images in the split
    :param splits_sdr: N lists, indices of HDR images in the split (may be len=0, if split contains only HDR)
    :param splits_hdr: N lists, indices of HDR images in the split (may be len=0, if split contains only SDR)
    :param split_names:
    :param run_tag:
    :return:
    """
    num_repeats_test = global_config["num_repeats_test"]
    original_checkpoint = global_config["load_checkpoint_file"]
    model_name = global_config["model"]
    pu_norm_scheme = "pu0_2.3" if pu_wrapper_config_base["normalize_pu_range_srgb"] else "pu0_1"
    output_dir = f"./output/{int(time.time())}-{run_tag}-{model_name}-{pu_norm_scheme}"
    if use_domain_adaptation:
        output_dir += "-da"
    output_file = "results.txt"
    os.makedirs(output_dir, exist_ok=True)
    logger = FileLogger("{}/{}".format(output_dir, output_file), verbose=True)

    results = []
    for split_num in range(n_splits):
        # reset this to the original
        global_config["load_checkpoint_file"] = original_checkpoint
        global_config["num_repeats_test"] = num_repeats_test

        split_name = split_names[split_num]
        # if "narwaria" not in split_name.lower():
        #     continue
        if model_name in models_pieapp:
            global_config["num_repeats_test"] = 1
        else:
            # for vtamiq, do more runs for HDR subsets
            split_is_hdr = str(split_name).lower() in ["narwaria", "korshunov", "hdr"]
            global_config["num_repeats_test"] *= 2 if split_is_hdr else 1

        set_split_fold(splits_full, splits_sdr, splits_hdr, split_num)

        logger(f"Starting run {split_num}: {split_name}...")
        global_config["output_tag"] = f"Run-{split_name}"
        global_config["output_dir"] = output_dir  # tell the script to write to cross-validation folder
        try:
            if use_domain_adaptation:
                # set_split_run_datasets_da(use_domain_adaptation=True)
                # first use domain adaptation
                result = train.train_domain_adaptation()
                logger(f"Finished DA run {split_num}: {split_name}:\n", result)
            else:
                # set_split_run_datasets_da(use_domain_adaptation=False)
                result = train.train()
                logger(f"Finished run {split_num}: {split_name}:\n", result)
            results.append(result)
        except Exception as e:
            logger(f"run_splits() exception during run for split_name={split_name}:", e)
            print("Continuing past exception...")

    for result in results:
        logger(result)

    return results, logger


def train_test_model_on_upiq(model_name, checkpoint):
    """
    This function evaluates models on UPIQ and includes training and testing on subsets of UPIQ.
    Note that the model should be initially pre-trained on sRGB data.
    UPIQ is split by cross-validation (60-20-20 splits) and across datasets (LIVE, TID2013, Korshunov, Narwaria).
    """

    upiq_common_settings()

    global_config["model"] = model_name

    run_id = int(time.time())

    for normalize_pu_range_srgb in [True, False]:
        pu_wrapper_config_base["normalize_pu_range_srgb"] = normalize_pu_range_srgb

        # TYPE 1: 5-fold cross-validation
        cross_val_type = f"CrossVal-{run_id}"
        global_config["load_checkpoint_file"] = checkpoint  # reset checkpoint
        n_folds = 5
        _, splits, splits_sdr, splits_hdr = upiq_get_split_indices(n_folds=n_folds, seed=0)
        split_names = list(range(n_folds))  # simply the fold count numbers
        results, logger = run_n_splits(False, n_folds, splits, splits_sdr, splits_hdr, split_names, cross_val_type)
        parse_runs(results, logger)  # print average results for cross-validation

        # TYPE 2: cross-dataset
        cross_val_type = f"CrossDataset-{run_id}"
        global_config["load_checkpoint_file"] = checkpoint  # reset checkpoint
        n_splits, splits, splits_sdr, splits_hdr, split_names = upiq_get_split_test_dataset()
        run_n_splits(False, n_splits, splits, splits_sdr, splits_hdr, split_names, cross_val_type)


# This function evaluates models on UPIQ without training on UPIQ.
# Note that the model should be initially pre-trained on sRGB data.
# UPIQ is split by cross-validation (60-20-20 splits) and across datasets (LIVE, TID2013, Korshunov, Narwaria).
def test_model_on_upiq(model_name, checkpoint, normalize_pu_range_srgb):

    upiq_common_settings()

    global_config["load_checkpoint_file"] = checkpoint
    global_config["model"] = model_name

    global_config["output_tag"] = "sRGB-pre-trained"

    # IMPORTANT: USE INDICES and not full dataset!
    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES
    global_config["allow_use_full_dataset"] = False

    global_config["dataset"] = DATASET_UPIQ_HDR_FULL
    global_config["num_repeats_test"] = 4

    pu_wrapper_config_base["normalize_pu_range_srgb"] = normalize_pu_range_srgb

    # NOTE: only testing will run, since train/val is disabled
    global_config["do_train"] = False  # disabled training
    global_config["do_val"] = False  # disabled validation
    global_config["do_test"] = True  # enabled testing

    # TYPE 1: 5-fold cross-validation
    run_tag = "CrossVal-sRGB"
    n_folds = 5
    _, splits, splits_sdr, splits_hdr = upiq_get_split_indices(n_folds=n_folds, seed=0)
    split_names = list(range(n_folds))  # simply the fold count numbers
    results, logger = run_n_splits(False, n_folds, splits, splits_sdr, splits_hdr, split_names, run_tag)
    parse_runs(results, logger)  # print average results for cross-validation

    # TYPE 2: Cross-dataset
    run_tag = "CrossDatasetTest-sRGB"
    n_folds, splits, splits_sdr, splits_hdr, split_names = upiq_get_split_test_dataset(
        split_sdr_datasets=True, split_hdr_datasets=True)
    run_n_splits(False, n_folds, splits, splits_sdr, splits_hdr, split_names, run_tag)


########################################################################################################################


def run_cross_val_da(da_type, model_name, model_checkpoint, coral_weight, normalize_pu_range_srgb):
    # global_config["is_debug"] = True
    # global_config["dataloader_num_workers"] = 1

    upiq_common_settings()

    global_config["model"] = model_name
    global_config["load_checkpoint_file"] = model_checkpoint

    global_config["use_pu"] = True
    pu_wrapper_config_base["normalize_pu_range_srgb"] = normalize_pu_range_srgb

    global_config["optimizer_learning_rate"] = 0.0001 if model_checkpoint is None else 0.00005
    global_config["scheduler_type"] = "lambda"
    global_config["scheduler_step_per_batch"] = True

    global_config["num_repeats_test"] = 4  # average over N runs for the test set (~ensemble)

    domain_adaptation_config["loss_source_weight"] = 1.0
    domain_adaptation_config["loss_target_weight"] = 1.0
    domain_adaptation_config["loss_coral_weight"] = coral_weight  # CORAL loss weight

    global_config["allow_use_full_dataset"] = True
    global_config["allow_use_full_dataset_test"] = False

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = True

    # LABELED HDR DA
    if da_type == DA_TYPE_S_HL:
        global_config["dataset"] = DATASET_UPIQ_HDR  # target
        domain_adaptation_config["dataset_source"] = DATASET_KADID10K  # source
        domain_adaptation_config['num_repeats_target'] = 8
        domain_adaptation_config["source_use_full_split"] = True  # use the entire dataset for source
        global_config["num_epochs"] = 30 if model_checkpoint else 15

    # UNLABELED HDR AND LABELED HDR-like
    elif da_type in [DA_TYPE_S_HU_SIHDR, DA_TYPE_S_HU_KADID, DA_TYPE_S_HU_UPIQS, DA_TYPE_S_HS]:
        global_config["allow_use_full_dataset"] = True
        global_config["do_test"] = False  # test will be done separately

        if da_type in [DA_TYPE_S_HU_SIHDR, DA_TYPE_S_HU_KADID, DA_TYPE_S_HU_UPIQS]:
            domain_adaptation_config["dataset_source"] = DATASET_KADID10K  # source
            if da_type == DA_TYPE_S_HU_SIHDR:
                global_config["dataset"] = DATASET_SIHDR
            if da_type == DA_TYPE_S_HU_KADID:
                global_config["dataset"] = DATASET_KADID10K
            if da_type == DA_TYPE_S_HU_UPIQS:
                global_config["dataset"] = DATASET_UPIQ_SDR2HDR

            global_config["do_val"] = False  # no validation for unlabeled data

            domain_adaptation_config["loss_target_weight"] = 0.0  # disable loss on unlabeled data

            # NOTE: for SI-HDR and UPIQ, there is roughly 10 times more SDR data than HDR data
            # optionally, target dataset can be repeated to approximately match the source dataset length
            domain_adaptation_config['num_repeats_target'] = 1

            global_config["num_epochs"] = 30 if model_checkpoint is None else 6

        elif da_type == DA_TYPE_S_HS:
            global_config["do_val"] = False
            global_config["dataset"] = DATASET_KADID10K  # target
            domain_adaptation_config["dataset_source"] = DATASET_KADID10K  # source

            global_config["num_epochs"] = 30 if model_checkpoint is None else 4

        domain_adaptation_config["source_use_full_split"] = True  # use the entire dataset for source

        pu_wrapper_config_display_ambient_random["randomize_ambient"] = True
        pu_wrapper_config_display_ambient_random["randomize_L_max"] = True

        if da_type != DA_TYPE_S_HU_UPIQS:
            # set custom display model for target datasets SIHDR and KADID
            domain_adaptation_config["pu_wrapper_config_target"] = {
                "display_L_max": 5000,
                "display_L_cr": 50000,
                "rand_L_max_delta": 500,
                "rand_distrib_ambient_mean": 100,
                "rand_distrib_ambient_delta": 25,
            }

    else:
        raise NotImplementedError()

    domain_adaptation_config['use_feats_separate'] = False
    domain_adaptation_config['use_feats_diff'] = False  # diff signal
    domain_adaptation_config['use_feats_dist'] = True and dataset_target() != DATASET_SIHDR  # dist images

    assert not (global_config["dataset"] == DATASET_SIHDR and
                (domain_adaptation_config['use_feats_diff'] or domain_adaptation_config['use_feats_dist'])), \
        f"Dataset {DATASET_SIHDR} has no distorted images."

    use_domain_adaptation = True

    # LABELED HDR DA
    if da_type == DA_TYPE_S_HL:
        cross_val_tag = f"CrossVal-HDR-{da_type}-coral{float2str3(domain_adaptation_config['loss_coral_weight'])}"
        n_folds = 5
        _, splits, splits_sdr, splits_hdr = upiq_get_split_indices(n_folds=n_folds, seed=0)
        split_names = list(range(n_folds))  # simply the fold count numbers
        results, logger = run_n_splits(
            use_domain_adaptation, n_folds, splits, splits_sdr, splits_hdr, split_names, cross_val_tag)
        parse_runs(results, logger)  # print average results for cross-validation

    # UNLABELED HDR AND LABELED HDR-LIKE DATA
    elif da_type in [DA_TYPE_S_HU_SIHDR, DA_TYPE_S_HU_KADID, DA_TYPE_S_HU_UPIQS, DA_TYPE_S_HS]:
        global_config["output_tag"] = da_type
        train.train_domain_adaptation()
        # # update with checkpoint from the DA run
        global_config["load_checkpoint_file"] = f"{global_config['output_dir_final']}/latest.pth"
        global_config["config_validated"] = False  # reset config validation flag

        # only enable test runs
        global_config["do_train"] = False
        global_config["do_val"] = False
        global_config["do_test"] = True

        domain_adaptation_config["pu_wrapper_config_target"] = None  # reset to default
        global_config["dataset"] = DATASET_UPIQ_HDR_FULL  # target train
        global_config["dataset_test"] = DATASET_UPIQ_HDR_FULL  # target test

        use_domain_adaptation = False  # disable DA
        cross_val_type = f"CrossDataset-HDR-{da_type}-coral{float2str3(domain_adaptation_config['loss_coral_weight'])}"
        n_folds, splits, splits_sdr, splits_hdr, split_names = upiq_get_split_test_dataset(
            # select only the HDR split
            split_sdr_datasets=False, split_hdr_datasets=False, split_sdr=False, split_hdr=True
        )
        run_n_splits(use_domain_adaptation, n_folds, splits, splits_sdr, splits_hdr, split_names, cross_val_type)

    else:
        raise ValueError()
