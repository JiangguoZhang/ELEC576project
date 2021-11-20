import random

import numpy as np
import pandas as pd
import torch
from PIL import ImageOps
from torchvision import datasets


class PairedNeurons(datasets.ImageFolder):
    def __init__(self, root, rle_dir, crop_size=256, is_train=True):
        self.masks = pd.read_csv(rle_dir)
        self.crop_size = crop_size
        self.is_train = is_train
        super().__init__(root)

    @staticmethod
    def rle_decode(mask_rle, shape, color=True):
        s = mask_rle.split()
        starts = list(map(lambda x: int(x) - 1, s[0::2]))
        lengths = list(map(int, s[1::2]))
        ends = [x + y for x, y in zip(starts, lengths)]
        img = np.zeros((shape[0] * shape[1]), dtype=bool)
        for start, end in zip(starts, ends):
            img[start: end] = color
        return img.reshape(shape)

    def __getitem__(self, index):
        img0, target = super().__getitem__(index)
        target = self.classes[index]
        labels = self.masks[self.masks["id"] == target]["annotation"].tolist()
        img1 = np.zeros((520, 704), dtype=bool)
        for label in labels:
            img1 = np.bitwise_or(img1, self.rle_decode(label, (520, 704)))
        img1 = np.array(img1, dtype=np.float32) * 2 - 1
        img0 = ImageOps.grayscale(img0)
        img0 = np.array(img0, dtype=np.float32) / 255 * 2 - 1

        if self.is_train:
            img_shape = np.shape(img0)
            x = random.randint(0, img_shape[0] - self.crop_size)
            y = random.randint(0, img_shape[1] - self.crop_size)
            is_flip = random.random() < 0.5  # flop
            r_angle = np.random.randint(0, 4)  # rotate
            img0 = img0[x:(self.crop_size + x), y:(self.crop_size + y)]
            img1 = img1[x:(self.crop_size + x), y:(self.crop_size + y)]
            if is_flip:
                img0 = np.fliplr(img0)
                img1 = np.fliplr(img1)
            img0 = np.rot90(img0, k=r_angle)
            img1 = np.rot90(img1, k=r_angle)

        img0 = torch.from_numpy(img0[np.newaxis, :, :].copy())
        img1 = torch.from_numpy(img1[np.newaxis, :, :].copy())

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img0, img1, target
