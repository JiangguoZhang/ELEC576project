import shutil
import os


def separate_test(train_dir, test_dir, test_samples):
    for items in os.listdir(test_samples):
        shutil.move(os.path.join(train_dir, items), os.path.join(test_dir, items))


separate_test("/mnt/data/elec576/save_method3/train", "/mnt/data/elec576/save_method3/test", "/mnt/data/elec576/project/1213/IMGS/test_stat")