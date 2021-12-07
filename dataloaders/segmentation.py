from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
from xlwt import Workbook
from skimage.segmentation import clear_border
SMOOTH = 1e-6
def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(2)
    
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()

img_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train"
csv_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, 256, is_train=False)
sum1,sum2,sum3,sum4,sum5,sum6=0,0,0,0,0,0
# Workbook is created
wb = Workbook()
  
# add_sheet is used to create sheet.

sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0,0,"Image Name")
sheet1.write(0,1,"IOU for Binary+OSTU")
sheet1.write(0,2,"segmented further using watershed")
sheet1.write(0,3,"Using distance transform and thresholding")
sheet1.write(0,4,"threshold the dist transform at 1/2 its max value.")
for i in range(len(pn)):
    x, y, l = pn.__getitem__(i)
    
    sheet1.write(i+1, 0, l)

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # # print(fig.shape)
    # # print(y.shape)
    # # fig.colorbar(im)
    # plt.savefig(os.path.join("./save", l))
    # plt.close()
    # plt.subplot(2, 3, i + 1)
    ###1
    x=np.uint8(x*255)
    axs[0,0].imshow(y, cmap="gray")
    axs[0,0].axis("off")
    axs[0,0].title.set_text("Grouth truth seg")
    
    ret, th1 = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel, iterations = 1)
    opening = clear_border(opening) #Remove edge touching grains
    # print(iou_numpy(x.astype(int),np.uint8(opening).astype(int)))
    sum1+=iou_numpy(x,opening.astype(int))
    sheet1.write(i+1,1,sum1)
    
    
    axs[0,1].imshow(opening, cmap="gray")
    axs[0,1].axis("off")
    axs[0,1].title.set_text("Threshold image to binary using OTSU")
    ###2
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    axs[0,2].imshow(sure_bg, cmap="gray")
    axs[0,2].axis("off")
    axs[0,2].title.set_text("segmented further using watershed")
    sum2+=iou_numpy(np.uint8(y*255),sure_bg.astype(int))
    sheet1.write(i+1,2,sum2)
    ###
    ###3
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    axs[1,0].imshow(dist_transform, cmap="gray")
    axs[1,0].axis("off")
    axs[1,0].title.set_text("Using distance transform and thresholding")
    sum3+=iou_numpy(np.uint8(y*255),dist_transform.astype(int))
    sheet1.write(i+1,3,sum3)
    ###4
    ret2, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    axs[1,1].imshow(sure_bg, cmap="gray")
    axs[1,1].axis("off")
    axs[1,1].title.set_text("threshold the dist transform at 1/2 its max value.")
    sum4+=iou_numpy(np.uint8(y*255),sure_bg.astype(int))
    sheet1.write(i+1,4,sum4)
    ####
    
    
    ###5 Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
    unknown = cv2.subtract(sure_bg,sure_fg)
    sum4+=iou_numpy(np.uint8(y*255),sure_bg.astype(int))
    axs[1,2].imshow(unknown, cmap="gray")
    axs[1,2].axis("off")
    axs[1,2].title.set_text("Unknown ambiguous region is nothing but bkground ")
    sheet1.write(i+1,5,sum5)
    
    
    fig.tight_layout()
    # print(iou_numpy((x*255).astype(int),(th1*255).astype(int)))
    plt.savefig(os.path.join("./save", l))
    plt.close()
    # plt.title(l)
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(opening, 'gray')
# plt.show()
wb.save('result.xls')
# print(sum/len(pn))