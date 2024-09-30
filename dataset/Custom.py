import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Union, Iterable
from PIL import Image
from torchvision import datasets, transforms
import yaml

class ImageDataset(Dataset):
    def __init__(self, path: Union[str, Path, Iterable], suffix: Iterable[str] = ("png", "jpg"),
                 mode: str = "RGB", transform=None):
        super().__init__()

        if isinstance(path, str) or isinstance(path, Path):
            path = [path]

        self.images = []
        for m in suffix:
            # support for multiple folders of data
            for p in path:
                p = Path(p)
                self.images += list(p.glob(f"*.{m}"))

        if mode not in ["RGB", "L", "CMYK"]:
            raise ValueError("mode must be one of {'RGB', 'L', 'CMYK'}")
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = str(self.images[item])
        image = Image.open(image_path).convert(self.mode)

        if self.transform is not None:
            image = self.transform(image)
        # Returns a useless label to correspond to the mnist and cifar dataset format
        return image, torch.zeros(1)


def create_custom_dataset(data_path, batch_size, **kwargs):
    norm = (0.5,) if kwargs.get("mode", "RGB") == "L" else (0.5, 0.5, 0.5)
    image_size = kwargs.get("image_size", (256, 256))
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm, norm)
    ])
    dataset_param = dict(
        suffix=kwargs.get("suffix", ("png", "jpg")),
        mode=kwargs.get("mode", "RGB"),
    )
    dataset = ImageDataset(data_path, transform=trans, **dataset_param)

    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)
    return dataloader


def create_custom_dataset_with_imagefolder(data_path, batch_size, **kwargs):
    norm = (0.5,) if kwargs.get("mode", "RGB") == "L" else (0.5, 0.5, 0.5)
    image_size = kwargs.get("image_size", (256, 256))

    # Define transformations
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm, norm)
    ])

    # Use ImageFolder to create the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=trans)

    # DataLoader params
    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)
    return len(dataset.classes), dataloader

# 测试总的类别数量
# def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
#     if isinstance(yml_path, str):
#         yml_path = Path(yml_path)
#     with yml_path.open('r', encoding=encoding) as f:
#         cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
#         return cfg
#
# if __name__ =='__main__':
#     config = load_yaml("config.yml", encoding="utf-8")
#     print(create_custom_dataset_with_imagefolder(**config["Dataset"]))