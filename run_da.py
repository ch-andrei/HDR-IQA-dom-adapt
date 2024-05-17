import argparse

from run_pretrain_model import parse_model_name
from training.train_da import *


# to acquire pre-trained models, see run_pretrain_model.py
checkpoint_vtamiq = "./output/vtamiq_srgb.pth"
checkpoint_pieapp = "./output/pieapp_srgb.pth"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_da.py',
        description='This script enters training with domain adaptation between SDR and HDR.\n'
    )
    parser.add_argument('model_name', type=str, help='model to be trained: "vtamiq" or "pieapp"')
    parser.add_argument('da_type', type=str,
                        help='domain adaptation type, select from: '
                             '[S_HU_SIHDR, S_HU_KADID, S_HU_UPIQS, S_HS, S_HL] '
                             '(not case sensitive))')
    parser.add_argument('--coral', type=float, default=0.025,
                        help="CORAL loss weight lambda (recommended 0.01 < lambda < 0.1)")
    parser.add_argument('--scratch', action='store_true',
                        help="train models from scratch instead of fine-tuning pre-trained (untested)")

    args = parser.parse_args()

    model_name = parse_model_name(args.model_name)

    if model_name == MODEL_VTAMIQ:
        checkpoint = checkpoint_vtamiq
    elif model_name == MODEL_PIEAPP:
        checkpoint = checkpoint_pieapp
    else:
        raise NotImplementedError()

    if args.scratch:
        checkpoint = None

    da_type = args.da_type.upper()
    coral_weight = float(args.coral)

    # NOTE: 0-2.3 normalization (normalize_pu_range_srgb=True) tends to have inconsistent results
    run_cross_val_da(da_type, model_name, checkpoint, coral_weight, normalize_pu_range_srgb=False)
