import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def get_iou(mask,predict):
    """mask: [0,1] numpy"""
    SMOOTH = 1e-6

    #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
    predict = np.heaviside(predict-0.5, 1.0)
    
    predict = predict.astype(np.int16)
    image_mask = mask.astype(np.int16)

    intersection = (predict & image_mask).sum((0, 1))
    union = (predict | image_mask).sum((0, 1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    print('iou=%f' % iou)

    return iou

def get_dice(mask,predict):
    """mask: [0,1], numpy"""
    #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
    predict = np.heaviside(predict-0.5, 1.0)

    predict = predict.astype(np.int16)
    image_mask = mask.astype(np.int16)

    intersection = (predict&image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice

def get_hd(mask,predict):
    """mask: [0,1]"""
    #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
    predict = np.heaviside(predict-0.5, 1.0)
    
    hd1 = directed_hausdorff(mask, predict)[0]
    hd2 = directed_hausdorff(predict, mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res


def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()