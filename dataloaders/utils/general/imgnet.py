import os
import shutil

def imgnet_move(img_dir):
    for img_name in os.listdir(img_dir):
        if img_name.endswith(".png"):
            new_dir = os.path.join(img_dir, img_name.split(".")[0])
            os.makedirs(new_dir)
            shutil.move(os.path.join(img_dir, img_name), os.path.join(new_dir, img_name))


imgnet_move("/home/derek/Disk1/cell_instance_segment/train")