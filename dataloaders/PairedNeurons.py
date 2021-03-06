import random
import os
import numpy as np
import pandas as pd
import torch
from PIL import ImageOps, Image
from torchvision import datasets, transforms
import cv2
import math
from skimage.segmentation import clear_border
from scipy import ndimage
import matplotlib.pyplot as plt

class PairedNeurons(datasets.ImageFolder):
    def __init__(self, root, rle_dir, crop_x=256, crop_y=256, norm_min=-1, norm_max=1, is_train=True,
                 is_supervised=True):
        if is_supervised:
            self.masks = pd.read_csv(rle_dir)
        else:
            self.masks = None
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.is_train = is_train
        self.is_supervised = is_supervised
        self.x_length = 520
        self.y_length = 704
        self.crop_diag = np.sqrt(np.square(crop_x) + np.square(crop_y))
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

    def rotate_images(self, images, angle):
        # Get the image size
        image = images[0]
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
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        right_bound, top_bound = np.max(rotated_coords, axis=0)
        left_bound, bot_bound = np.min(rotated_coords, axis=0)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        rotated_coords = np.array(rotated_coords) + np.array([new_w, new_h]) // 2   # Change to new coordinates
        right_idx, top_idx = np.argmax(rotated_coords, axis=0)
        left_idx, bot_idx = np.argmin(rotated_coords, axis=0)
        coords_list = [rotated_coords[left_idx], rotated_coords[bot_idx],   # Get the vertices in anticlockwise order
                       rotated_coords[right_idx], rotated_coords[top_idx]]  # from the left-most vertex

        # Compute the transform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        results = [cv2.warpAffine(
            image_transform,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        ) for image_transform in images]

        whole_size = results[0].shape   # The new image size

        iterations = 0
        while True:
            if iterations > 100:
                x_coord = (whole_size[0] - self.crop_x) // 2
                y_coord = (whole_size[1] - self.crop_y) // 2
                break
            x_coord = np.random.randint(whole_size[0] - self.crop_x)
            y_coord = np.random.randint(whole_size[1] - self.crop_y)
            points = np.array([[x_coord, y_coord],
                              [x_coord + self.crop_x, y_coord],
                              [x_coord + self.crop_x, y_coord + self.crop_y],
                              [x_coord, y_coord + self.crop_y]])
            are_points_in_rect = self.is_point_in_rect(coords_list, points)
            if are_points_in_rect:
                break
            iterations += 1

        y_coord = whole_size[1] - y_coord - self.crop_y  # The y axis is flipped
        # x_coord = whole_size[0] - x_coord - self.crop_x
        return [result[x_coord:x_coord+self.crop_x, y_coord:y_coord+self.crop_y] for result in results]

    def rotate_images2(self, images, angle):
        # Get the image size
        image = images[0]
        image_size = image.shape
        new_len = np.max([abs(self.crop_diag * np.cos((angle + 45) / 180 * np.pi)),
                         abs(self.crop_diag * np.sin((angle + 45) / 180 * np.pi))])
        new_len = int(np.ceil(new_len))
        x_sample = np.random.randint(new_len//2, image_size[0] - np.ceil(new_len/2))
        y_sample = np.random.randint(new_len//2, image_size[1] - np.ceil(new_len/2))

        x_min = min([x_sample, 0])
        y_min = min([y_sample, 0])

        rot_imgs = [ndimage.rotate(img[x_min:x_min+new_len, y_min:y_min+new_len], angle, reshape=True) for img in images]
        rot_img_shape = rot_imgs[0].shape
        img_center = [rot_img_shape[0] // 2, rot_img_shape[1] // 2]
        return [img[img_center[0] - self.crop_x // 2:img_center[0] + self.crop_x // 2,
                img_center[1] - self.crop_y // 2:img_center[1] + self.crop_y // 2] for img in rot_imgs]

    @staticmethod
    def is_point_in_rect(coord_list, points):
        """
        Check if all points of a small rectangle in the large rectangle coordinate list
        :param coord_list: Large rectangle coordinate list ordered counterclockwise from the left most
        :param points: Vertices of the small rectangle ordered counterclockwise from the bottom left
        :return: Is small rectangle in large rectangle
        """
        cross_value = [np.cross(coord_list[(i + 1) % 4]-coord_list[i], points[i]-coord_list[i]) for i in range(4)]
        cross_value = np.array(cross_value)
        if np.all(cross_value > 0):
            return True
        else:
            return False

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

        if width > image_size[0]:
            width = image_size[0]

        if height > image_size[1]:
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    @staticmethod
    def segmentation(image):
        """
        Using Otsu's threshold to segment image
        :param image: image for segmentation
        :return: Normalized segmented image
        """
        image = np.uint8((image / 255 * 2 - 1) * 255)
        ret, th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
        opening = clear_border(opening)
        sure_bg = cv2.dilate(opening, kernel, iterations=4)
        return sure_bg

    def remove_padding(self, image):
        p_img_shape = image.shape
        x_pad = (p_img_shape[0] - self.x_length) // 2
        y_pad = (p_img_shape[1] - self.y_length) // 2
        return image[x_pad:x_pad+self.x_length, y_pad:y_pad+self.y_length]

    @staticmethod
    def rle_encode(mask):
        """
        mask: numpy array binary mask
        1 - mask
        0 - background
        Returns encoded run length
        """
        rle = []
        mask_height, mask_width = np.shape(mask)
        for i in range(mask_height):
            pixels = mask[i, :]
            pixels = np.concatenate([[0], pixels, [0]])
            runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs = runs + i * mask_width
            runs[1::2] -= runs[::2]
            rle.append(' '.join(str(x) for x in runs))
        return ' '.join(rle)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img0 = self.loader(path)
        if self.transform is not None:
            img0 = self.transform(img0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        pathname = os.path.basename(path)
        target_name = pathname.split(".")[0]
        img0 = ImageOps.grayscale(img0)
        img0 = np.array(img0)
        img_shape = img0.shape
        self.x_length, self.y_length = img_shape
        if self.is_supervised:
            img1 = np.zeros(img_shape, dtype=bool)
            labels = self.masks[self.masks["id"] == target_name]["annotation"].tolist()
            for label in labels:
                img1 = np.bitwise_or(img1, self.rle_decode(label, img_shape))
            img1 = np.uint8(img1 * 255)
        else:
            img1 = self.segmentation(img0)

        # img0 np [0,255], img1 np [0, 255]
        if self.is_train:
            # Process images in numpy format
            img0, img1 = self.rotate_images2([img0, img1], angle=float(np.random.random(1) * 360 - 180))
            is_flip = random.random() < 0.5  # flip
            if is_flip:
                img0 = np.fliplr(img0)
                img1 = np.fliplr(img1)
        else:
            # Process images in PIL format
            img0 = Image.fromarray(img0)
            img1 = Image.fromarray(img1)
            if self.crop_x > img_shape[0]:
                pad_x = (self.crop_x - img_shape[0]) // 2
            else:
                pad_x = 0
            if self.crop_y > img_shape[1]:
                pad_y = (self.crop_y - img_shape[1]) // 2
            else:
                pad_y = 0
            padding = transforms.Pad(padding=(pad_y, pad_x), padding_mode='reflect')
            img0 = padding(img0)
            img1 = padding(img1)
            img0 = np.array(img0)
            img1 = np.array(img1)

        img0 = img0 / 255 * (self.norm_max - self.norm_min) + self.norm_min
        img1 = img1 / 255 * (self.norm_max - self.norm_min) + self.norm_min

        img0 = torch.FloatTensor(img0[np.newaxis, :, :].copy())
        img1 = torch.FloatTensor(img1[np.newaxis, :, :].copy())

        return img0, img1, target_name
