import random

import numpy as np
import pandas as pd
import torch
from PIL import ImageOps
from torchvision import datasets
import cv2
import math
import matplotlib.pyplot as plt

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

    @staticmethod
    def rotate_image(image, angle):
        # Get the image size
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    @staticmethod
    def largest_rotated_rect(w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi
        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)
        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    @staticmethod
    def crop_around_center(image, width, height):

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def __getitem__(self, index):
        img0, target = super().__getitem__(index)
        target = self.classes[index]
        labels = self.masks[self.masks["id"] == target]["annotation"].tolist()
        img1 = np.zeros((520, 704), dtype=bool)
        for label in labels:
            img1 = np.bitwise_or(img1, self.rle_decode(label, (520, 704)))
        # no need to convert to [-1,1], just remain [0, 1]
        # img1 = np.array(img1, dtype=np.float32) * 2 - 1
        img1 = np.array(img1, dtype=np.float32)
        
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
        else:
            img_shape = np.shape(img0)
            x = random.randint(0, img_shape[0] - self.crop_size)
            y = random.randint(0, img_shape[1] - self.crop_size)
            img0 = img0[x:(self.crop_size + x), y:(self.crop_size + y)]
            img1 = img1[x:(self.crop_size + x), y:(self.crop_size + y)]

        img0 = torch.from_numpy(img0[np.newaxis, :, :].copy())
        img0 = img0.repeat(3,1,1) # convert to 3 channels, for unet
        img1 = torch.from_numpy(img1[np.newaxis, :, :].copy())

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img0, img1, target
