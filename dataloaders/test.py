from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import os
img_dir = "/Users/harry/Desktop/ELEC576project/sartorius-cell-instance-segmentation/train"
csv_dir = "/Users/harry/Desktop/ELEC576project/sartorius-cell-instance-segmentation/train.csv"

pn = PairedNeurons(img_dir, csv_dir, 256, is_train=False)
for i in range(len(pn)):
    x, y, l = pn.__getitem__(i)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    im = axs[0].imshow(x, cmap="gray")
    axs[0].axis("off")
    axs[1].imshow(y, cmap="gray")
    axs[1].axis("off")
    # fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(os.path.join("/Users/harry/Desktop/ELEC576project/save", l))
    plt.close()
