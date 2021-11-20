from collections import OrderedDict

import torch.nn as nn

from .Blocks import ConstantBlock, AdaIN, NoiseLayer, Linear, Convolution, Deconvolution, Normalization, Activation, \
    Dropout2d, Pad, Downscale
from .Calculations import Reshape, Reparameterize, Concat, Addition


def build_sequential_layer(info_dict, on_gpu=False):
    layer_keys = list(map(int, info_dict.keys()))
    layer_keys.sort()
    layers = []
    for layer_key in layer_keys:
        layer_info = info_dict[str(layer_key)]
        name = layer_info["name"]
        if layer_info["type"] == "sequential":
            new_layer = build_sequential_layer(layer_info["params"], on_gpu=on_gpu)
        else:
            new_layer = get_layer(layer_info, on_gpu=on_gpu)
        layers.append((name, new_layer))
    return nn.Sequential(OrderedDict(layers))


def get_layer(info, on_gpu=False):
    layer = None
    params = info["params"] if "params" in info else None
    if info["type"] == "conv" or info["type"] == "encode":
        layer = Convolution(params["num_in"], params["num_out"], params["kernel_size"],
                            stride=params["stride"], padding=params["padding"], dilation=params["dilation"],
                            bias=bool(params["bias"]))
    elif info["type"] == "deconv":
        layer = Deconvolution(params["num_in"], params["num_out"], params["kernel_size"],
                              stride=params["stride"], padding=params["padding"],
                              dilation=params["dilation"], output_padding=params["output_padding"])
    elif info["type"] == "linear":
        layer = Linear(params["num_in"], params["num_out"])
    elif info["type"] == "noise":
        layer = NoiseLayer(params["num_in"])
    elif info["type"] == "AdaIN":
        layer = AdaIN(params["num_in"], params["num_out"])
    elif info["type"] == "const":
        layer = ConstantBlock(params["num_in"], params["img_size"])
    elif info["type"] == "reshape":
        layer = Reshape(params["depth"])
    elif info["type"] == "reparameterize":
        layer = Reparameterize(on_gpu=on_gpu)
    elif info["type"] == "concat":
        layer = Concat(params["axis"])
    elif info["type"] == "normalization":
        layer = Normalization(params["type"], params["num_out"])
    elif info["type"] == "activation":
        if "negative_slope" in params:
            layer = Activation(params["type"], negative_slope=params["negative_slope"])
        else:
            layer = Activation(params["type"])
    elif info["type"] == "dropout2d":
        layer = Dropout2d(params["probability"])
    elif info["type"] == "pad":
        layer = Pad(params["type"], params["padding"])
    elif info["type"] == "avgPool2d":
        layer = Downscale(params["kernel_size"], stride=params["stride"], padding=params["padding"],
                          count_include_pad=bool(params["count_include_pad"]))
    elif info["type"] == "addition":
        layer = Addition()

    return layer
