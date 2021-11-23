import cv2 as cv
import matplotlib.pyplot as plt

# 1.读取图像
img = cv.imread('/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train/0030fd0e6378/0030fd0e6378.png', 0)
# 2. 阈值分割
thresholdValue=135
ret, th1 = cv.threshold(img, thresholdValue, 255, cv.THRESH_BINARY)
ret,th2 = cv.threshold(img,200,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret, th4 = cv.threshold(img, thresholdValue, 255, cv.THRESH_TOZERO)

# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
sure_bg = cv2.dilate(closing,kernel,iterations=3)

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