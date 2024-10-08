from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def create_cifar10_dataset(data_path, batch_size, **kwargs):
    train = kwargs.get("train", True)
    download = kwargs.get("download", True)

    dataset = CIFAR10(root=data_path, train=train, download=download, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return len(dataset.classes), dataloader
