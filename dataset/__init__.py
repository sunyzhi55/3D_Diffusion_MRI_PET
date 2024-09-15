from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar10_dataset
from dataset.Custom import create_custom_dataset, create_custom_dataset_with_imagefolder
from dataset.Custom3D import create_custom3d_dataset, NiiDataset
import torch

def create_dataset(dataset: str, **kwargs):
    if dataset == "mnist":
        return create_mnist_dataset(**kwargs)
    elif dataset == "cifar":
        return create_cifar10_dataset(**kwargs)
    elif dataset == "custom":
        # return create_custom_dataset(**kwargs)
        return create_custom_dataset_with_imagefolder(**kwargs)
    elif dataset == "custom3D":
        data_path = kwargs['data_path']
        csv_file = kwargs['csv_file']
        batch_size = kwargs['batch_size']
        valid_group = kwargs['valid_group']
        dataset = NiiDataset(data_path, csv_file, valid_group=valid_group)
        return create_custom3d_dataset(dataset, batch_size)
    else:
        raise ValueError(f"dataset except one of {'mnist', 'cifar', 'custom', 'custom3D'}, but got {dataset}")
