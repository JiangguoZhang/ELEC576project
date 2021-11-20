import os
from tensorboardX import SummaryWriter
import torch


def pix2pixHD_G(model, logdir, comment, shape, use_cuda=False):
    if not os.path.exists(logdir):
        log = SummaryWriter(logdir, comment)
        with log:
            tmp_input = torch.rand(shape[0], shape[1], shape[2], shape[3])
            if use_cuda:
                tmp_input = tmp_input.cuda()
            log.add_graph(model, input_to_model=tmp_input)


def pix2pixHD_D(model, logdir, comment, shape, use_cuda=False):
    model.struct_log(logdir, comment, shape, use_cuda=use_cuda)


