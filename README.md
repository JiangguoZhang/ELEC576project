# ELEC576project - Semi-supervised Segmentation of High-resolution Neuronal Cells

## About This Repo
This repo, **Semi-supervised Segmentation of  High-resolution Neuronal Cells**, is mainly for the ELEC576 project. Here is a brief introduction of each file:
- Traditional method segmenation is in `ELEC576project/dataloaders/segmentation.py `
- Data augmentation method is implemented in `ELEC576project/dataloaders/PairedNeurons.py`
- Train the pix2pixHD model by running `ELEC576project/train_pix2pixHD.py`
- Test our model by running `ELEC576project/test_pix2pixHD.py`

# How to Run UNet Model
The only thing you should do is enter the dataset.py and correct the path of the datasets. Then run:
```python
python main.py --action 'train&test' --arch attention-unet --epoch 200 --batch_size 20 --dataset yhead
```

# Environment
This project depends on the following environments and packages:
- window10(Ubuntu is OK)
- python3.6
- pytorch1.3.1  

# Results
after train and test,3 folders will be created,they are "result","saved_model","saved_predict".

## saved_model folder:
After training,the saved model is in this folder.

## result folder:
in result folder,there are the logs and the line chart of metrics.such as:
![image](./UNET/linechart.png)

## saved_predict folder:
in this folder,there are the ouput predict of the saved model,such as:
![image](./seg_result.png)
