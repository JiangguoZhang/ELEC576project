from PairedNeurons import PairedNeurons
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

img_dir = "/mnt/data/elec576/project/1213/IMGS/unsup_stat"
csv_dir = "/mnt/data/elec576/project/1213/unsup_stat.csv"
threshold = 0.9

pn = PairedNeurons.rle_encode

example = np.array([[1,1,1,1,0],[0,0,1,0,1],[1,0,0,0,1],[1,1,0,0,0]])

id_list = []
annotation_list = []

for img_file in os.listdir(img_dir):
    img_name = img_file.split(".")[0]
    img_path = os.path.join(img_dir, img_file)
    img = np.load(img_path)
    img = img > threshold
    rle_encoded = pn(img)
    id_list.append(img_name)
    annotation_list.append(rle_encoded)

csv_df = pd.DataFrame({"id": id_list,
                       "annotation": annotation_list})

csv_df.to_csv(csv_dir, sep=',', encoding='utf-8')