import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def parser():
    """
        Provides a default argument parser including arguments required for
        most networks

        :returns: parser
    """
    par = argparse.ArgumentParser()
    par.add_argument('--batch-size', type=int, default=10, help='input batch size')
    par.add_argument('--save-every', type=int, default=10, help='save checkpoint every N epochs. 0=No save')
    par.add_argument('--no-cuda', action='store_false', help='disables cuda')
    par.add_argument('--epochs', type=int, default=1000, help='Number of train epochs')
    par.add_argument('--comment', default='', help='Tensorboard run comment')
    return par


def mnist(batch_size, download_loc, train=True, pin_memory=False, download=False):
    """
        A default MNIST dataset. Images are transofrmed to tensors with a range
        of 0 to 1,

        :param train: retun train or test dataset (default is train)
        :param download_loc: where to download the dataset to
        :param pin_memory: passed to torchvision.DataLoader
        :param download: download the dataset
        :returns: torchvision.DataLoader containing MNIST dataset
    """
    dataset = datasets.MNIST(download_loc, train=train, download=download,
                     transform=transforms.Compose([
                        transforms.ToTensor(),
                     ]))

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=pin_memory)

    return train_loader

def svhn(batch_size,download_loc,train=True,pin_memory=False,download=False):
    """
        A default grayscale SVHN dataset. Images are converted to grayscale,
        resized to 28 x 28, and transformed to a range of 0 to 1.

        :param train: retun train or test dataset (default is train)
        :param download_loc: where to download the dataset to
        :param pin_memory: passed to torchvision.DataLoader
        :param download: download the dataset
        :returns: torchvision.DataLoader containing SVHN dataset
    """

    if train:
        split = 'train'
    else:
        split = 'test'

    dataset = datasets.SVHN(download_loc, split=split, download=download,
                    transform=transforms.Compose([
                       transforms.Resize((28,28)),
                       transforms.Grayscale(),
                       transforms.ToTensor()
                    ]))

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=pin_memory)

    return train_loader
