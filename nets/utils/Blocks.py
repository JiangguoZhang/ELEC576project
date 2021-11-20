import torch
import torch.nn as nn
from .Calculations import PixelNorm


class ConstantBlock(nn.Module):
    """
    Create the constant block
    """

    def __init__(self, channels, size):
        super().__init__()
        self.const = nn.Parameter(torch.ones(1, channels, size[0], size[1]), requires_grad=True)

    def forward(self, x):
        return self.const.expand(x.size(0), -1, -1, -1)


class AdaIN(nn.Module):
    """
    Process AdaIN, the input should be a list of two items [image_x, latent_w]
    """

    def __init__(self, n_w, n_channel):
        super().__init__()
        self.style_transform = nn.Linear(n_w, n_channel*2)
        self.input_normalize = nn.InstanceNorm2d(n_channel)

    def forward(self, input_list):
        assert len(input_list) == 2, "Not a valid input, size = %d" % len(input_list)
        x, w = input_list
        x = self.input_normalize(x)
        style = self.style_transform(w)
        style_shape = [-1, 2, int(x.size(1))] + (x.dim() - 2) * [1]
        style = style.view(style_shape)  # [batch_size, 2, channel, 1, 1...]
        x = x * (style[:, 0] + 1.) + style[:, 1]  # x has been normalized, style[:,0]+1 is y_s
        return x


class NoiseLayer(nn.Module):
    """
    Add noise to the image
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels), requires_grad=True)   # Weight is trainable

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class BasicProcess(nn.Module):
    """
    Construct basic process, which is able to work on one or more inputs
    """
    def __init__(self, act=None):
        super().__init__()
        if act:
            self.act = act
        else:
            self.act = self.return_back

    @staticmethod
    def return_back(x):
        return x

    def forward(self, x):
        if isinstance(x, tuple):
            result = tuple(self.act(item) for item in x)
        else:
            result = self.act(x)
        return result


class Linear(BasicProcess):
    def __init__(self, num_in, num_out):
        super().__init__(nn.Linear(num_in, num_out))


class Convolution(BasicProcess):
    def __init__(self, num_in, num_out, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__(nn.Conv2d(num_in, num_out, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))


class Deconvolution(BasicProcess):
    def __init__(self, num_in, num_out, kernel_size, stride=1, padding=0, dilation=1, output_padding=0):
        super().__init__(nn.ConvTranspose2d(num_in, num_out, kernel_size, stride=stride, padding=padding,
                                            dilation=dilation, output_padding=output_padding))


class Normalization(BasicProcess):
    def __init__(self, name, channels):
        act = self.return_back
        if name == "BatchNorm2d":
            act = nn.BatchNorm2d(channels)
        elif name == "PixelNorm":
            act = PixelNorm()
        super().__init__(act)


class Activation(BasicProcess):
    def __init__(self, name, negative_slope=None):
        act = self.return_back
        if name == "LeakyReLU":
            if negative_slope:
                act = nn.LeakyReLU(negative_slope, True)
            else:
                act = nn.LeakyReLU(True)
        elif name == "Sigmoid":
            act = nn.Sigmoid()
        elif name == "ReLU":
            act = nn.ReLU(True)
        elif name == "Tanh":
            act = nn.Tanh()
        super().__init__(act)


class Dropout2d(BasicProcess):
    def __init__(self, probability):
        super().__init__(nn.Dropout2d(probability))


class Pad(BasicProcess):
    def __init__(self, name, padding):
        if name == "reflect":
            act = nn.ReflectionPad2d(padding)
        else:
            act = nn.ReplicationPad2d(padding)
        super().__init__(act)


class Downscale(BasicProcess):
    def __init__(self, kernel_size, stride, padding, count_include_pad):
        super().__init__(nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad))