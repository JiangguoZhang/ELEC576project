import cv2
import numpy as np

# Step1. 加载图像
img = cv2.imread(
    '/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train/0030fd0e6378/0030fd0e6378.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
