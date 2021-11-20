from tkinter import _flatten

import torch.nn as nn

from nets.utils import get_layer, build_sequential_layer

'''
    This is the network model that can load different network structures
'''


class ConvNet(nn.Module):
    #  Initializer
    def __init__(self, s, r_key, on_gpu=False):
        super(ConvNet, self).__init__()
        self.r_key = r_key
        self.on_gpu = on_gpu
        self.layers = []
        layer_keys = list(map(int, s.keys()))
        layer_keys.sort()
        self.keys = []
        for layer_key in layer_keys:
            layer = s[str(layer_key)]
            name = layer["name"]
            self.keys.append(name)
            transact = self.layer_builder(layer, on_gpu=self.on_gpu)
            feed_in = layer["input"] if isinstance(layer["input"], list) else [layer["input"]]
            self.layers.append({"type": layer["type"], "name": name, "feed_in": feed_in})
            # Since the parameters are not set as direct attributes, we have to manually register the parameters
            if transact:
                self.add_module(name, transact)

    # Build a layer
    @staticmethod
    def layer_builder(layer_info, on_gpu=False):
        if layer_info["type"] == "sequential":
            new_layer = build_sequential_layer(layer_info["params"], on_gpu=on_gpu)
        else:
            new_layer = get_layer(layer_info, on_gpu=on_gpu)
        return new_layer

    # Get the final output
    def forward(self, *x):
        outs = {"x": x}
        for layer in self.layers:
            if len(layer["feed_in"]) == 1:
                feed_in = outs[layer["feed_in"][0]]
                if isinstance(feed_in, tuple) and len(feed_in) == 1:
                    feed_in = feed_in[0]
            else:
                feed_in = [outs[feed] for feed in layer["feed_in"]]
                feed_in = _flatten(feed_in)  # indiv_conv can return to 2
            transact = self.__getattr__(layer["name"])
            outs[layer["name"]] = transact(feed_in)
        output = []
        if isinstance(self.r_key, list):
            for key_value in self.r_key:
                output.append(outs[key_value])
        else:
            output = outs[self.r_key]
        return output

