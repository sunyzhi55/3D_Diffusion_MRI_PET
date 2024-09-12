from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar10_dataset
from dataset.Custom import create_custom_dataset, create_custom_dataset_with_imagefolder
from dataset.Custom3D import load_dataset
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
        # return load_dataset(**kwargs)
        data_path = kwargs['data_path']
        return load_dataset(data_dir=data_path, device=torch.device('cpu'), augment = True, argument_side = 3)
    else:
        raise ValueError(f"dataset except one of {'mnist', 'cifar', 'custom', 'custom3D'}, but got {dataset}")
