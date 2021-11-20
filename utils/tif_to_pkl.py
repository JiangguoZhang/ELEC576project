import os
import argparse
import numpy as np
import pickle as pkl
import skimage.external.tifffile as tiffreader
from PIL import Image


class Downsample:
    def __init__(self, resize_by):
        self.resize_by = resize_by

    def execute(self, file_name):
        if file_name.endswith(".jpg"):
            image = Image.open(file_name).convert('I')
        else:
            image = tiffreader.imread(file_name)
            image = Image.fromarray(image).convert("I")
        img_size = np.array(image.size) * self.resize_by
        img_size = img_size.astype(int)
        image = image.resize(img_size.astype(int), Image.ANTIALIAS)
        return np.array(image, dtype=np.uint8)


def tif2pkl(tif_dir, pkl_dir, name, label, name_format, start_idx, end_idx, resize_by=0.5):
    image_stack = []
    label_list = []
    ds = Downsample(resize_by)

    for idx in range(start_idx, end_idx+1):
        image = ds.execute(os.path.join(tif_dir, name_format % idx))
        image_stack.append(image)
        label_list.append("%s_%d" % (name, idx))
    image_stack = np.array(image_stack)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    pkl.dump([image_stack, label_list], open(os.path.join(pkl_dir, "%s_%s.pkl" % (name, label)), 'wb'), protocol=4)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transform the .tif images to a .pkl stack.')
    parser.add_argument('--tif-dir', default='./imgs/', type=str, help='The directory of your .tif/.jpg images.')
    parser.add_argument('--pkl-dir', default='./data/hdt1_phase.pkl', type=str, help='The directory for .pkl file.')
    parser.add_argument('--name', default='ds1', type=str, help='The name of this dataset.')
    parser.add_argument('--label', default='phase', type=str, help='The label of the image stack.')
    parser.add_argument('--name-format', default='images_%d.tif', type=str, help='The specific format for images.')
    parser.add_argument('--start-idx', default=0, type=int, help='The start index.')
    parser.add_argument('--end-idx', default=100, type=int, help='The end index.')
    parser.add_argument('--resize-by', default=1, type=float, help='Resize the image for sufficient training.')
    args = parser.parse_args()

    tif2pkl(args.tif_dir, args.pkl_dir, args.name, args.label, args.name_format, args.start_idx, args.end_idx,
            resize_by=args.resize_by)
