import os
import argparse
import pickle as pkl
import skimage.external.tifffile as tiffreader


def pkl2tif(pkl_dir, tif_dir):
    img_stack = pkl.load(open(pkl_dir, "rb"))

    if not os.path.exists(tif_dir):
        os.makedirs(tif_dir)

    for i in range(len(img_stack[0])):
        img = img_stack[0][i, :, :]
        tiffreader.imsave(os.path.join(tif_dir, "image_%d.tif" % i), img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transform the .pkl image stack to individual .tif images.')
    parser.add_argument('--pkl-dir', default='./data/hdt1_phase.pkl', type=str,
                        help='The directory for .pkl file.')
    parser.add_argument('--tif-dir', default='./imgs/', type=str, help='The directory to save the images.')
    args = parser.parse_args()

    pkl2tif(args.pkl_dir, args.tif_dir)