from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
SMOOTH = 1e-6
def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(2)
    
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded.mean()  # Or thresholded.mean()

img_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train"
csv_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, 256, is_train=False)
sum=0
for i in range(len(pn)):
    x, y, l = pn.__getitem__(i)

    # fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # # print(fig.shape)
    # # print(y.shape)
    # im = axs[0].imshow(y, cmap="gray")
    # axs[0].axis("off")
    # axs[1].imshow(y, cmap="gray")
    # axs[1].axis("off")
    # # fig.colorbar(im)
    # fig.tight_layout()
    # plt.savefig(os.path.join("./save", l))
    # plt.close()
    # plt.subplot(2, 3, i + 1)
    thresholdValue=0.07
    ret, th1 = cv2.threshold(x, thresholdValue, 0.1, cv2.THRESH_BINARY)
    # print(iou_numpy((x*255).astype(int),(th1*255).astype(int)))
    sum+=iou_numpy((x*255).astype(int),(th1*255).astype(int))
    # plt.title(l)
    # plt.imshow(x*255, 'gray')
print(sum/len(pn))