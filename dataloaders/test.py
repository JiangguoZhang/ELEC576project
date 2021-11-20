from dataloaders import PairedNeurons
from matplotlib import pyplot as plt

img_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train"
csv_dir = "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv"
pn = PairedNeurons(img_dir, csv_dir, 256)

x, y = pn.__getitem__(25)

fig, axs = plt.subplots(1, 2)

im = axs[0].imshow(x, cmap="gray")
axs[1].imshow(y, cmap="gray")

fig.colorbar(im)

plt.show()

i = 1
