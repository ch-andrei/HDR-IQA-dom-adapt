from utils.logging import log_warn
from torchvision import models as tv
from torchvision.models import VGG16_Weights


def vgg_features_to_pieapp(use_pupieapp_naming=True):
    log_warn("torchvision warnings about 'pretrained' and 'weights' params can be ignored...")

    vgg16_state_dict = tv.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.state_dict()

    # NOTE: to match PieAPP structure. some layers are repeated
    layer_match_dict = {
        "conv1.weight": "0.weight",
        "conv1.bias": "0.bias",
        "conv2.weight": "2.weight",
        "conv2.bias": "2.bias",
        "conv3.weight": "2.weight",  # repeating layer 7
        "conv3.bias": "2.bias",
        "conv4.weight": "5.weight",
        "conv4.bias": "5.bias",
        "conv5.weight": "7.weight",
        "conv5.bias": "7.bias",
        "conv6.weight": "7.weight",
        "conv6.bias": "7.bias",  # repeating layer 7
        "conv7.weight": "10.weight",
        "conv7.bias": "10.bias",
        "conv8.weight": "12.weight",
        "conv8.bias": "12.bias",
        "conv9.weight": "14.weight",
        "conv9.bias": "14.bias",
        "conv10.weight": "17.weight",
        "conv10.bias": "17.bias",
        "conv11.weight": "19.weight",
        "conv11.bias": "19.bias",
    }

    # build pieapp state dict from vgg state dict
    pieapp_features_state_dict = {}
    for layer_name in layer_match_dict:
        # PU-Pieapp uses .extractor and .comparitor modules instead of all conv layers directly in model
        layer_name_pieapp = ("extractor." if use_pupieapp_naming else "") + layer_name
        pieapp_features_state_dict[layer_name_pieapp] = vgg16_state_dict[layer_match_dict[layer_name]]
    return pieapp_features_state_dict
