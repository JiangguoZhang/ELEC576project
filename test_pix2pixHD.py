import argparse
import io
import json
import os
import pickle as pkl

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib_scalebar.scalebar import ScaleBar
from torch.utils.data import DataLoader

import dataloaders
from dataloaders.utils.general import NormalizeTif
from nets import GAN2D
from pytorch_utils import util, batcher

EPS = 1e-8

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=6, help='input batch size')
parser.add_argument('--no-cuda', action='store_false', help='disables cuda')
parser.add_argument('--net-struct', default='./structure/pix2pixHD.json',
                    help='The net structure.')
parser.add_argument('--multiGPU', action='store_true',
                    help='''Enable training on multiple GPUs, uses all that are available.''')
parser.add_argument('--dataset-loc',
                    default="/mnt/data/elec576/project/test",
                    help='Folder containing training dataset')
parser.add_argument('--csv-loc',
                    default="/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv",
                    help='The csv file of rle masks')
common_dir = "IMGS"
scratch_dir = "1207-semi1"
parser.add_argument('--load',
                    default=scratch_dir + '/ckpts/CHECKPOINT-400',
                    help='''Load pre-trained networks''')
parser.add_argument('--img-loc',
                    default='%s/%s/test_img' % (scratch_dir, common_dir),
                    help='The directory to save output images.')
parser.add_argument('--stat-loc',
                    default='%s/%s/test_stat' % (scratch_dir, common_dir),
                    help='The analysis results are saved in this directory.')

parser.add_argument('--visualize', action='store_true', help='Visualize result.')
parser.add_argument('--movie', action='store_true', help='Create movie result.')
parser.add_argument('--norm-min', type=int, default=-1, help="The normalized minimum")
parser.add_argument('--norm-max', type=int, default=1, help="The normalized maximum")
parser.add_argument('--num-workers', type=int, default=3, help="The number of cores to load images.")
parser.add_argument('--crop-x', type=int, default=720, help='The height of input image.')
parser.add_argument('--crop-y', type=int, default=720, help='The width of input image.')
parser.add_argument('--crop-size', type=int, default=512, help='The input size.')
parser.add_argument('--g1', default="g1_out", help='The name of the final layer in generator 1.')
parser.add_argument('--g2', default="g2_out", help='The name of the final layer in generator 2.')
parser.add_argument('--d-layer', default="feat", help='The name of the final layer in discriminator.')
parser.add_argument('--n-layers', type=int, default=3, help='The levels of discriminator.')
opt = parser.parse_args()
#opt.no_cuda = False
opt.visualize = True
print(opt)
if not os.path.exists(opt.img_loc):
    os.makedirs(opt.img_loc)

if not os.path.exists(opt.stat_loc):
    os.makedirs(opt.stat_loc)

s = json.load(open(opt.net_struct, "rb"))
G = GAN2D.ConvNet(s["G"], [opt.g1, opt.g2]).eval()

dataset = dataloaders.PairedNeurons(opt.dataset_loc, opt.csv_loc, crop_x=opt.crop_x, crop_y=opt.crop_y,
                              norm_min=opt.norm_min, norm_max=opt.norm_max, is_train=False, is_supervised=True)
train_loader = DataLoader(
    dataset,
    num_workers=opt.num_workers,  # Use this to replace data_prefetcher
    batch_size=opt.batch_size,
    shuffle=False,
    pin_memory=opt.no_cuda
)

if opt.multiGPU:
    G = nn.DataParallel(G)

'''Load net from directory'''
if opt.load is not None:
    util.load_nets(opt.load, {
                       'G': G
                   }, on_gpu=opt.no_cuda)

if opt.no_cuda:
    G = G.cuda()

batch = batcher()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def test(norm_rage=None):
    imgList = []
    with torch.no_grad():
        for batch_idx, tl in enumerate(train_loader):
            img0, img1, target = tl
            print("\r Processing the %d th batch." % batch_idx, end='')
            batch.batch()
            if opt.no_cuda:
                img0 = img0.cuda()
            _, x_fake = G(img0)
            # pkl.dump(img0.cpu().data.numpy(), open("0810/tmp.pkl", 'wb'))
            im0 = img0.cpu().data.numpy()
            im1 = img1.cpu().data.numpy()
            im_gen = x_fake.cpu().data.numpy()
            for i in range(np.size(im1, axis=0)):
                ground_truth = dataset.remove_padding(im1[i, 0, :, :])
                gen = dataset.remove_padding(im_gen[i, 0, :, :])
                neuron = dataset.remove_padding(im0[i, 0, :, :])
                save_dir = os.path.join(opt.stat_loc, target[i])
                np.save(save_dir, gen)

                if opt.visualize:
                    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
                    ax[0].imshow(neuron, cmap='gray')
                    ax[0].axis('off')
                    ax[0].set_title("Neuron")
                    #ax[0].text(10, 50, "A", fontsize=36, color='white')

                    ax[1].imshow(ground_truth, cmap='gray')
                    ax[1].axis('off')
                    ax[1].set_title("Ground Truth")
                    #ax[1].text(10, 50, "B", fontsize=36, color='white')

                    ax[2].imshow(gen, cmap='gray')
                    ax[2].axis('off')
                    ax[2].set_title("Synthesized")
                    #ax[2].text(10, 50, "C", fontsize=36, color='white')

                    fig.tight_layout()
                    scalebar = ScaleBar(1.3e-6, location="upper right")  # 1 pixel = 1.3 microns
                    plt.gca().add_artist(scalebar)
                    img_dir = opt.img_loc
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    plt.savefig(os.path.join(img_dir, "%s.png" % target[i]))
                    if opt.movie:
                        img = get_img_from_fig(fig)
                        imgList.append(img)
                    plt.close(fig)


    if opt.movie:
        movieSize = imgList[0].shape[:2]
        movieSize = (movieSize[1], movieSize[0])
        out = cv2.VideoWriter(os.path.join(opt.img_loc, "movie.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 4, movieSize)
        for img in imgList:
            out.write(img)
        out.release()


test()