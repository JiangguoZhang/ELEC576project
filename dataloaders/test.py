from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import os
img_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train"
csv_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, crop_x=520, crop_y=704, is_train=False, is_supervised=False)
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
