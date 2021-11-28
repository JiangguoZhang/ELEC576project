import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
SMOOTH = 1e-6
def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(2)
    
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded.mean()  # Or thresholded.mean()
# 1.读取图像
img = cv.imread('/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train/0030fd0e6378/0030fd0e6378.png',0)
# 2. 阈值分割
thresholdValue=135
ret, th1 = cv.threshold(img, thresholdValue, 255, cv.THRESH_BINARY)
ret,th2 = cv.threshold(img,200,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret, th4 = cv.threshold(img, thresholdValue, 255, cv.THRESH_TOZERO)


# 3. 图像显示
titles = ['original', 'th1', 'th2', 'th3', 'th4', 'th5']
images = [img, th1,th2, th4]
plt.figure(figsize=(10,6))
# 使用Matplotlib显示
for i in range(4):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.show()