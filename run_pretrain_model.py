import argparse

from training.train_config import *
from training.train import train


def parse_model_name(model_name):
    model_name = model_name.lower()
    if model_name == "vtamiq":
        return MODEL_VTAMIQ
    elif model_name == "pieapp":
        return MODEL_PIEAPP
    else:
        raise ValueError()


def train_model(model_name):
    global_config["model"] = model_name
    global_config["load_checkpoint_file"] = None  # train from scratch

    # only enable training (will use the full dataset, no val/test splits)
    global_config["do_train"] = True
    global_config["do_val"] = False
    global_config["do_test"] = False

    global_config["dataset"] = DATASET_KADID10K

    global_config["num_epochs"] = 25
    global_config["optimizer_learning_rate"] = 0.0001
    global_config["scheduler_type"] = "lambda"

    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_pretrain_model_srgb.py',
        description='This script is used to pre-train VTAMIQ and PieAPP on KADID10k.\n'
                    'ex: run with VTAMIQ as "run_pretrain_model_srgb.py vtamiq"'
    )
    parser.add_argument('model_name', type=str, help='model to be trained: "vtamiq" or "pieapp"')
    parser.add_argument('--pu', action='store_true',
                        help="train on PU-encoded data (will use display model and PU encoding)")
    parser.add_argument('--n255', action='store_true',
                        help="when running with --pu, training with --n255 will use 0-2.3 normalization "
                             "(will map PU-encoded SDR to 0-1 and PU-encoded HDR values to 1-2.3), "
                             "else 0-1 normalization, for PU-encoded data.")
    args = parser.parse_args()

    if args.pu:
        global_config["use_pu"] = True

    if args.n255:
        pu_wrapper_config_base["normalize_pu_range_srgb"] = True

    model_name = parse_model_name(args.model_name)

    train_model(model_name)
