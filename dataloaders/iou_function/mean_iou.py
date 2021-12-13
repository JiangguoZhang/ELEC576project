import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def rle_decode(mask_rle, shape, color=True):
    s = mask_rle.split()
    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]
    img = np.zeros((shape[0] * shape[1]), dtype=bool)
    for start, end in zip(starts, ends):
        img[start: end] = color
    return img.reshape(shape)


def iou(predict, label, threshold=0.5):
    predict = predict >= threshold
    TP = predict & label
    P_all = predict | label
    return np.sum(TP) / np.sum(P_all)


def mean_iou(pred_dir, label_csv, save_dir="/mnt/data/elec576/project/1120/compare"):
    labels = pd.read_csv(label_csv)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    acc_list = []
    for item in os.listdir(pred_dir):
        image_id = item.split(".")[0]
        pred_img = np.load(os.path.join(pred_dir, item))
        pred_img = (pred_img + np.min(pred_img)) / np.max(pred_img) - np.min(pred_img)
        rle_codes = labels[labels["id"] == image_id]["annotation"].tolist()
        img_shape = pred_img.shape
        label_img = np.zeros_like(pred_img, dtype=bool)
        for rle_code in rle_codes:
            label_img = np.bitwise_or(label_img, rle_decode(rle_code, img_shape))

        iou_list = [iou(pred_img, label_img, threshold=threshold) for threshold in np.arange(0.5, 1, 0.05)]
        iou_mean = np.mean(iou_list)
        #fig, axs = plt.subplots(1, 2, figsize=[12, 6])
        #axs[0].imshow(pred_img, cmap="gray")
        #axs[1].imshow(label_img, cmap="gray")
        #axs[1].set_title("%.2g, %.2g" % (iou_list[0], iou_list[-1]))
        #fig.tight_layout()
        #plt.savefig(os.path.join(save_dir, image_id))
        #plt.close()
        #print(iou_mean)
        acc_list.append(iou_mean)

    acc = np.mean(acc_list)
    print(acc)

mean_iou("/mnt/data/elec576/project/1207-semi1/IMGS/stat", "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv")
