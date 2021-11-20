import torch
import torch.nn as nn
from torch.autograd import Variable

class Reshape(nn.Module):

    def __init__(self, depth):
        super(Reshape, self).__init__()
        if isinstance(depth, list):
            self.depth = depth
            self.dims = len(depth)
        else:
            self.depth = [depth]
            self.dims = 1

    def forward(self, x):
        if self.dims == 1:
            return x.view(self.depth[0])
        elif self.dims == 2:
            return x.view(self.depth[0], self.depth[1])
        elif self.dims == 3:
            return x.view(self.depth[0], self.depth[1], self.depth[2])
        elif self.dims == 4:
            return x.view(self.depth[0], self.depth[1], self.depth[2], self.depth[3])


class Reparameterize(nn.Module):

    def __init__(self, on_gpu=True):
        super().__init__()
        self.on_gpu = on_gpu

    def forward(self, input_list):
        mu, logstd = input_list
        std = logstd.exp()
        if self.on_gpu:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps, requires_grad=True)

        return eps.mul(std).add_(mu)


class PixelNorm(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class Concat(nn.Module):

    def __init__(self, axis=3):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.cat(x, self.axis)
        

class Addition(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def addition(x):
        y = 0
        for item in x:
            y += item
        return y
        
    def forward(self, x):
        return self.addition(x)

