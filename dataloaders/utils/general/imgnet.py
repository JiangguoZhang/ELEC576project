import os
import pandas as pd
import shutil


def imgnet_move(img_dir, csv_dir):
    cell_types = pd.read_csv(csv_dir)
    ids = cell_types.id.unique()
    for index in ids:
        indiv_img_dir = os.path.join(img_dir, index + ".png")
        cell_type = cell_types[cell_types.id == index].cell_type.values[0]
        cell_type_dir = os.path.join(img_dir, cell_type)
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir)
        shutil.copy(indiv_img_dir, os.path.join(cell_type_dir, index + ".png"))


imgnet_move("/mnt/data/elec576/project/train",
            "/mnt/data/elec576/project/kaggle_cell_segmentation/sartorius-cell-instance-segmentation/train.csv")
