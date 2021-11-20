import datetime
import os
import random
from multiprocessing import Pool

import numpy as np
import pickle as pkl
import torch.utils.data

from dataloaders.utils.general import NormalizeTif


class MyxoLabelLoaderT(torch.utils.data.Dataset):
    """
    The MyxoLabelLoaderT load consecutive images from individual package each time
    This process will save memory
    """

    def __init__(self, root_dir, name_list, inputLabel, groundTruthLabel, seq_len=3, mode="train", rotate=0,
                 crop_size=256, batch_size=50, with_he=True, pre_process=False, one_set=False, pin_memory=False):
        self.root_dir = root_dir
        self.name_list = name_list
        self.inputLabel = inputLabel
        self.groundTruthLabel = groundTruthLabel
        self.seq_len = seq_len
        self.with_he = with_he
        self.pre_process = pre_process
        self.one_set = one_set
        self.pin_memory = pin_memory
        # Sort the package based on the index (format: image_%04d.pkl)
        self.he_method = NormalizeTif(norm_method=1, norm_range=[-1, 1])
        self.clahe_method = NormalizeTif(norm_method=2, norm_range=[-1, 1])
        self.input, self.targets, self.groundTruth, self.gt_he = [], [], [], []
        self.len_list = []
        self.set_idx = 0
        for name in name_list:
            ipt, iptl = pkl.load(open(os.path.join(root_dir, "%s_%s.pkl" % (name, inputLabel)), "rb"))
            gt, _ = pkl.load(open(os.path.join(root_dir, "%s_%s.pkl" % (name, groundTruthLabel)), "rb"))
            if self.pre_process:
                self.input.append(np.array([self.clahe_method.forward(ipt[idx, :, :]) for idx in range(len(ipt))]))
                self.groundTruth.append(np.array([self.clahe_method.forward(gt[idx, :, :]) for idx in range(len(gt))]))
                self.gt_he.append(np.array([self.he_method.forward(gt[idx, :, :]) for idx in range(len(gt))]))
            else:
                self.input.append(ipt)
                self.groundTruth.append(gt)
            self.targets.extend(iptl)
            self.len_list.append(len(ipt) // seq_len)
            if self.one_set:
                break
        self.len_list = np.cumsum(self.len_list)
        self.length = self.len_list[-1]
        self.mode = mode
        self.crop_size = crop_size
        self.rotate = rotate
        self.batch_size = batch_size


    def transforms(self, imgs, crop_size):
        '''
            Data Augmentation
            Augments moVIe by:
                A) performing a random windowed crop of size 128*128
                B) Randomly flip the movie
                C) Rotate image 0,90,180, or 270 Degrees randomly
                D) jitter the movie by +- max_fitter frames, padding with the first or
                   last frame as needed
                E) Randomly adjusting the contrast by a random value
                   between alpha=(min,max)

            Input movies must be normalized to between 0,1.

            loosely based off the work at https://github.com/aleju/imgaug
        '''
        imgs_trans = []
        if self.mode == "train":
            # Window Crop
            img_shape = np.shape(imgs[0])[-2:]
            x = random.randint(0, img_shape[0] - crop_size)
            y = random.randint(0, img_shape[1] - crop_size)
            is_flip = random.random() < 0.5  # flop
            r_angle = np.random.randint(0, 4)  # rotate
            for i in range(len(imgs)):
                img_tmp = None
                if imgs[i] is not None:
                    img_tmp = imgs[i][:, x:(crop_size + x), y:(crop_size + y)]
                    if is_flip:
                        img_tmp = np.fliplr(img_tmp)
                    img_tmp = np.rot90(img_tmp, k=r_angle, axes=(1, 2))
                    img_tmp = torch.from_numpy(img_tmp.copy())
                imgs_trans.append(img_tmp)
        else:
            for i in range(len(imgs)):
                img_tmp = None
                if imgs[i] is not None:
                    img_tmp = imgs[i][:, :self.crop_size, :self.crop_size]
                    if self.rotate:
                        img_tmp = np.rot90(img_tmp, k=self.rotate, axes=(1, 2))
                    img_tmp = torch.from_numpy(img_tmp.copy())
                imgs_trans.append(img_tmp)
        return imgs_trans

    def next_set(self):
        self.set_idx = (self.set_idx + 1) % len(self.name_list)
        name = self.name_list[self.set_idx]
        self.input, self.targets, self.groundTruth, self.gt_he = [], [], [], []
        self.len_list = []
        ipt, iptl = pkl.load(open(os.path.join(self.root_dir, "%s_%s.pkl" % (name, self.inputLabel)), "rb"))
        gt, _ = pkl.load(open(os.path.join(self.root_dir, "%s_%s.pkl" % (name, self.groundTruthLabel)), "rb"))
        if self.pre_process:
            self.input.append(np.array([self.clahe_method.forward(ipt[idx, :, :]) for idx in range(len(ipt))]))
            self.groundTruth.append(np.array([self.clahe_method.forward(gt[idx, :, :]) for idx in range(len(gt))]))
            self.gt_he.append(np.array([self.he_method.forward(gt[idx, :, :]) for idx in range(len(gt))]))
        else:
            self.input.append(ipt)
            self.groundTruth.append(gt)
        self.targets.extend(iptl)
        self.len_list.append(len(ipt) // self.seq_len)
        self.len_list = np.cumsum(self.len_list)
        self.length = self.len_list[-1]
        return 0

    def __getitem__(self, index):
        index = index * self.seq_len
        list_idx = len(self.len_list) - sum(index < self.len_list)
        img_idx = index
        if list_idx > 0:
            img_idx = index - self.len_list[list_idx - 1]
        img1 = self.input[list_idx][img_idx:img_idx+self.seq_len, :, :]
        img2 = self.groundTruth[list_idx][img_idx:img_idx+self.seq_len, :, :]
        he_img = None
        target = self.targets[index:index+self.seq_len]
        if self.pre_process:
            input_img = img1
            gt_img = img2
            if self.with_he:
                he_img = self.gt_he[list_idx][img_idx:img_idx+self.seq_len, :, :]
        else:
            input_img = np.array([self.clahe_method.forward(img1[idx, :, :]) for idx in range(len(img1))])
            gt_img = np.array([self.clahe_method.forward(img2[idx, :, :]) for idx in range(len(img2))])
            if self.with_he:
                he_img = np.array([self.he_method.forward(img2[idx, :, :]) for idx in range(len(img2))])
        input_img, gt_img, he_img = self.transforms([input_img, gt_img, he_img], self.crop_size)
        if self.one_set and index == self.length - 1:  # Reach the end of one set
            self.next_set()
        if self.with_he:
            return input_img, gt_img, he_img, target
        else:
            return input_img, gt_img, target

    def __len__(self):
        return self.length
