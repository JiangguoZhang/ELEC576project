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
parser.add_argument('--net-struct', default='./structure/pix2pixHD_he-2.json',
                    help='The net structure.')
parser.add_argument('--multiGPU', action='store_true',
                    help='''Enable training on multiple GPUs, uses all that are available.''')
parser.add_argument('--dataset_loc',
                    default="data",
                    help='Folder containing dataset')
parser.add_argument('--name-list',
                    default="hr7",
                    help='The name sequence split by _')
common_dir = "0712"
scratch_dir = "test_results"
parser.add_argument('--load',
                    default='CHECKPOINTS',
                    help='''Load pre-trained networks''')
parser.add_argument('--img-loc',
                    default='%s/%s/img' % (scratch_dir, common_dir),
                    help='The directory to save output images.')
parser.add_argument('--stat-loc',
                    default='%s/%s/stat' % (scratch_dir, common_dir),
                    help='The analysis results are saved in this directory.')

parser.add_argument('--label1', default='phase', help='The input image label.')
parser.add_argument('--label2', default='tdTomato', help='The ground truth image label.')
parser.add_argument('--visualize', action='store_true', help='Visualize result.')
parser.add_argument('--movie', action='store_true', help='Create movie result.')
parser.add_argument('--num-workers', type=int, default=8, help="The number of cores to load images.")
parser.add_argument('--crop-size', type=int, default=1008, help='The input size.')
parser.add_argument('--g1', default="g1_out", help='The name of the final layer in generator 1.')
parser.add_argument('--g2', default="g2_out", help='The name of the final layer in generator 2.')
parser.add_argument('--g1-he', default="g1_heout", help='The name of the HE output layer in generator 1.')
parser.add_argument('--g2-he', default="g2_heout", help='The name of the HE output layer in generator 2.')
parser.add_argument('--d-layer', default="feat", help='The name of the final layer in discriminator.')
parser.add_argument('--n-layers', type=int, default=3, help='The levels of discriminator.')
opt = parser.parse_args()
#opt.no_cuda = False
opt.visualize = True
opt.name_list = opt.name_list.split("_")
print(opt)
if not os.path.exists(opt.img_loc):
    os.makedirs(opt.img_loc)

if not os.path.exists(opt.stat_loc):
    os.makedirs(opt.stat_loc)
    os.makedirs(opt.stat_loc + "2")

s = json.load(open(opt.net_struct, "rb"))
G = GAN2D.ConvNet(s["G"], [opt.g1, opt.g2, opt.g1_he, opt.g2_he]).eval()

train_loader = DataLoader(
    dataloaders.MyxoLabelLoaderT(opt.dataset_loc, opt.name_list, opt.label1, opt.label2, seq_len=1,
                                 mode="test", pre_process=False, crop_size=opt.crop_size),
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
    if not norm_rage:
        norm_rage = [-1, 1]
    #MSE_list = []
    #MSE_he_list = []
    #ssim_list = []
    #ssim_he_list = []
    imgList = []
    method = NormalizeTif(norm_range=[-1, 1], norm_method=1)
    with torch.no_grad():
        for batch_idx, tl in enumerate(train_loader):
            img0, img1, img2, target = tl
            print("\r Processing the %d th batch." % batch_idx, end='')
            batch.batch()
            if opt.no_cuda:
                img0 = img0.cuda()
            _, x_fake, _, x_fake_he = G(img0)
            # pkl.dump(img0.cpu().data.numpy(), open("0810/tmp.pkl", 'wb'))
            im0 = img0.cpu().data.numpy()
            im1 = img1.cpu().data.numpy()
            im_gen = x_fake.cpu().data.numpy()
            im_gen_he = x_fake_he.cpu().data.numpy()
            for i in range(np.size(im1, axis=0)):
                tdt = im1[i, 0, :, :]
                gen = im_gen[i, 0, :, :]
                #MSE = np.square(tdt - gen).mean()
                #MSE_list.append(MSE)
                #ssim_item = ssim(tdt, gen)
                #ssim_list.append(ssim_item)
                #tdt = np.uint16((tdt - norm_rage[0]) / (norm_rage[1] - norm_rage[0]) * 65535)
                #tdt = method.forward(tdt)
                #gen = np.uint16((gen - norm_rage[0]) / (norm_rage[1] - norm_rage[0]) * 65535)
                #gen = method.forward(gen)
                #MSE = np.square(tdt - gen).mean()
                #MSE_he_list.append(MSE)
                #ssim_item = ssim(tdt, gen)
                #ssim_he_list.append(ssim_item)
                phase = im0[i, 0, :, :]
                gen_he = im_gen_he[i, 0, :, :]
                result = {
                    "phase": phase,
                    "tdTomato": tdt,
                    "gen": gen,
                    "gen_he": gen_he
                }
                save_dir = os.path.join(opt.stat_loc, target[0][i].split('_')[0])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pkl.dump(result, open(os.path.join(save_dir, "pkg_%s.pkl" % target[0][i].split('_')[1]), "wb"))

                if opt.visualize:
                    fig, ax = plt.subplots(2, 3, figsize=[12, 8])
                    ax[0, 0].imshow(phase, cmap='gray')
                    ax[0, 0].axis('off')
                    ax[0, 0].set_title("phase contrast + CLAHE")
                    ax[0, 0].text(10, 50, "A", fontsize=36, color='white')

                    ax[0, 1].imshow(tdt, cmap='gray')
                    ax[0, 1].axis('off')
                    ax[0, 1].set_title("tdTomato + CLAHE")
                    ax[0, 1].text(10, 50, "B", fontsize=36, color='white')

                    ax[0, 2].imshow(gen, cmap='gray')
                    ax[0, 2].axis('off')
                    ax[0, 2].set_title("CLAHE output")
                    ax[0, 2].text(10, 50, "C", fontsize=36, color='white')

                    ax[1, 0].imshow(gen_he, cmap='gray')
                    ax[1, 0].axis('off')
                    ax[1, 0].set_title("HE output")
                    ax[1, 0].text(10, 50, "D", fontsize=36, color='white')
                    tdt = np.uint16((tdt - norm_rage[0]) / (norm_rage[1] - norm_rage[0]) * 65535)
                    tdt = method.forward(tdt)
                    ax[1, 1].imshow(tdt, cmap='gray')
                    ax[1, 1].axis('off')
                    ax[1, 1].set_title("tdTomato + CLAHE + HE")
                    ax[1, 1].text(10, 50, "E", fontsize=36, color='white')
                    gen = np.uint16((gen - norm_rage[0]) / (norm_rage[1] - norm_rage[0]) * 65535)
                    gen = method.forward(gen)
                    ax[1, 2].imshow(gen, cmap='gray')
                    ax[1, 2].axis('off')
                    ax[1, 2].set_title("CLAHE output + HE")
                    ax[1, 2].text(10, 50, "F", fontsize=36, color='white')

                    fig.tight_layout()
                    scalebar = ScaleBar(1.3e-6, location="upper right")  # 1 pixel = 1.3 microns
                    plt.gca().add_artist(scalebar)
                    img_dir = os.path.join(opt.img_loc, target[0][i].split('_')[0])
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    plt.savefig(os.path.join(img_dir, "image_%s.png" % target[0][i].split('_')[1]))
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