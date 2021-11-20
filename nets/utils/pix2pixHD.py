import torch
import torch.nn as nn
from torch.autograd import Variable
from nets import GAN2D
import os
from tensorboardX import SummaryWriter
from dataloaders.utils.general import Hook


class Pix2pixHDLoss(nn.Module):
    """
    This class is built to compare the discriminated result with a matrix.
    Reference: git@github.com:NVIDIA/pix2pixHD.git
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, use_cuda=False):
        """
        This function only calculate the loss on the last layer
        """
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.loss = nn.MSELoss()
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.use_cuda = use_cuda

    def get_target_tensor(self, input_loss, target_is_real):
        """
        Generate an all-real-label matrix for real case or an all-fake-label matrix for fake case
        :param input_loss: The input loss, whose size our target tensor size should match with
        :param target_is_real: True - if target is real; False - if target is fake.
        """
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input_loss.numel()))
            if create_label:
                real_tensor = self.Tensor(input_loss.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)    # These tensors are not trainable
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input_loss.numel()))
            if create_label:
                fake_tensor = self.Tensor(input_loss.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        if self.use_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input_loss, target_is_real):
        if isinstance(input_loss, list):
            loss = 0
            for loss_level in input_loss:   # The losses in Di
                if isinstance(loss_level, list):    # If we have features for different discriminator
                    target_tensor = self.get_target_tensor(loss_level[-1], target_is_real)
                    loss += self.loss(loss_level[-1], target_tensor)
                else:
                    target_tensor = self.get_target_tensor(loss_level, target_is_real)
                    loss += self.loss(loss_level, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_loss, target_is_real)
            return self.loss(input_loss, target_tensor)


class MultiLayerDiscriminator(nn.Module):
    def __init__(self, structure, out_key, levels=1, mode="train", para=False, all_keys=True):
        super().__init__()
        self.levels = levels
        self.hook = []
        for i in range(levels):
            # Build a MultiLayerDiscriminator
            model = GAN2D.ConvNet(structure, out_key)
            if mode == "train":
                model = model.train()
            self.add_module("D%d" % i, model)
            # Initialize hook
            self.hook.append(Hook())
        self.all_keys = all_keys
        self.start_level = 0
        self.para = para

    @staticmethod
    def get_D_in(stack_input, num_ds=1):
        D_in = stack_input
        if num_ds > 0:
            downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
            for i in range(num_ds):
                D_in = downsample(D_in)
        return D_in

    def set_hook(self):
        for idx_model in range(self.start_level, self.levels):
            model = self.__getattr__("D%d" % idx_model)
            if self.para:
                net_chilren = model.module.children()
            else:
                net_chilren = model.children()
            for child in net_chilren:
                child.register_forward_hook(hook=self.hook[idx_model].hook)

    def forward(self, stack_input):
        """
        Generate a list of feature lists, which contain the features of different layers for a specific discriminator.
        The input stack is also added to the front of each feature list.
        :param stack_input: The input stack
        :return d_mats: Model feature list, ordered by the index of discriminator.
        """
        d_mats = []  # Initialize the model feature list
        for num_ds, idx_model in enumerate(range(self.start_level, self.levels)):
            model = self.__getattr__("D%d" % idx_model)
            d_in = self.get_D_in(stack_input, num_ds=num_ds)
            d_mat = [d_in]  # Initialize the feature list with the input stack
            self.hook[idx_model].clear()
            res = model(d_in)
            if self.all_keys:   # If we want to get all features in the discriminator
                d_mat.extend(self.hook[idx_model].feature)  # Extend a list
            else:   # If we only want the output of discriminator
                d_mat.append(res)   # Append a single value
            d_mats.append(d_mat)
        return d_mats

    def parallel(self):
        for idx_model in range(self.start_level, self.levels):
            model = self.__getattr__("D%d" % idx_model)
            model = nn.DataParallel(model)
            self.__setattr__("D%d" % idx_model, model)
        self.para = True

    def struct_log(self, log_dir, comment, shape, use_cuda=False):
        """
        This function is built to draw the model structures in tensorboard
        :param log_dir: log dir
        :param comment: comment
        :param shape: shape
        :param use_cuda: use CUDA
        """
        for i in range(self.levels):
            present_logdir = log_dir + "%d" % i
            if not os.path.exists(present_logdir):
                log = SummaryWriter(present_logdir, comment)
                with log:  # All the inputs should be set with log
                    input_real = torch.rand(shape[0], shape[1], shape[2], shape[3])
                    input_gen = torch.rand(shape[0], shape[1], shape[2], shape[3])
                    model = self.__getattr__("D%d" % i)
                    stack_input = torch.cat([input_real, input_gen], dim=1)
                    if use_cuda:
                        stack_input = stack_input.cuda()
                    d_in = self.get_D_in(stack_input, num_ds=i)
                    log.add_graph(model, input_to_model=d_in)
        return 0


class FeatLoss(nn.Module):
    def __init__(self, n_layers_D=3, num_D=3, lambda_feat=1., criterionFeat=nn.L1Loss(), start_level=0):
        super().__init__()
        self.num_D = num_D
        self.D_weights = 1.0 / num_D
        self.feat_weights = 4.0 / (n_layers_D + 1)
        self.lambda_feat = lambda_feat
        self.criterionFeat = criterionFeat
        self.start_level = start_level

    def forward(self, pred_fake, pred_real):
        loss_G_GAN_Feat = 0.
        for i in range(self.num_D - self.start_level):  # If start_level is larger, there are fewer losses
            for j in range(len(pred_fake[i]) - 1):  # Do not calculate feature loss on the last layer
                loss_G_GAN_Feat += self.D_weights * self.feat_weights * \
                                   self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat
        return loss_G_GAN_Feat


def temporal_loss(img2, step):
    t_loss = torch.nn.MSELoss()
    img_length = img2.shape[0]
    numerator = 0
    for i in range(step-1):
        numerator += t_loss(img2[range(i, img_length, step), :], img2[range(i+1, img_length, step), :])
    #denominator = t_loss(img1[1:, :], img1[:-1, :])
    #constant = 1e-3
    return numerator #/ (denominator + constant)

