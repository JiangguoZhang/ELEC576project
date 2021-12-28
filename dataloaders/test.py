from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import numpy as np
import os

img_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train"
csv_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, crop_x=256, crop_y=256, norm_min=0, norm_max=1, is_train=True, is_supervised=True)

example = np.array([[1,1,1,1,0],[0,0,1,0,1],[1,0,0,0,1],[1,1,0,0,0]])
rle_encoded = pn.rle_encode(example)
print(rle_encoded)

"""
for i in range(len(pn)):
    x, y, l = pn.__getitem__(i)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    im = axs[0].imshow(x, cmap="gray")
    # axs[0].axis("off")
    axs[1].imshow(y, cmap="gray")
    #axs[1].axis("off")

    # fig.colorbar(im)
    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join("/mnt/data/elec576/project/dataloaders/save/train", l))
    plt.close()
"""