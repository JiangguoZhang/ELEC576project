from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import os
import cv2
img_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train"
csv_dir = "/Users/mac/Desktop/Rice-COMP576/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, 256, is_train=False)
for i in range(6):
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
    plt.subplot(2, 3, i + 1)
    plt.title(l)
    plt.imshow(y, 'gray')
plt.show()