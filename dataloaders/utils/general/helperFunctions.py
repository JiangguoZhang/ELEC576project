import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable


def calc_gradient_penalty(model, real_data, fake_data, conditional_data, use_cuda=False):
    '''
        Copied from https://github.com/caogang/wgan-gp
    '''
    LAMBDA = 10
    BATCH_SIZE = real_data.size()[0]

    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view_as(real_data)
    alpha = alpha.cuda() if use_cuda else alpha
    alpha = Variable(alpha, requires_grad=True)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = model(conditional_data, interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    gradient_penalty = ((grad_norm - 1) ** 2).mean() * LAMBDA

    return gradient_penalty, grad_norm


# Register forward hook
class Hook:
    def __init__(self):
        self.feature = []

    def clear(self):
        self.feature = []

    def hook(self, module, fea_in, fea_out):
        self.feature.append(fea_out)
        return None


