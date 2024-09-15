# import nibabel as nib
# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from pathlib import Path
# from skimage import transform as skt
#
# # 自定义 Dataset 类来处理 MRI 和 PET 数据
# class MriPetDataset(Dataset):
#     def __init__(self, mri_dir, pet_dir, csv_file, transform=None, argument_side=3, valid_group=("AD", "CN")):
#         """
#         Args:
#             mri_dir (string or Path): MRI 文件所在的文件夹路径。
#             pet_dir (string or Path): PET 文件所在的文件夹路径。
#             csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
#             transform (callable, optional): 可选的转换操作，应用于样本。
#         """
#         self.mri_dir = Path(mri_dir)
#         self.pet_dir = Path(pet_dir)
#         self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
#         self.transform = transform
#         self.argument_side = argument_side
#         self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 1, 'sSCD': 0, 'pSCD': 1,
#                        'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0, 'sCN': 0,
#                        'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
#         self.valid_group = valid_group
#
#         # 过滤只保留 valid_group 中的有效数据
#         self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()
#
#     def __len__(self):
#         return len(self.filtered_indices)
#
#     def __getitem__(self, idx):
#         # 获取过滤后的索引
#         filtered_idx = self.filtered_indices[idx]
#
#         # 获取对应的文件名和标签
#         img_name = self.labels_df.iloc[filtered_idx, 0]
#         label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签
#
#         # MRI 文件路径
#         mri_img_path = self.mri_dir / (img_name + '.nii')
#         mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
#
#         # PET 文件路径
#         pet_img_path = self.pet_dir / (img_name + '.nii')
#         pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
#
#         # 对 MRI 和 PET 数据应用相同的转换
#         if self.transform:
#             mri_img_numpy = self.transform(mri_img_numpy)
#             pet_img_numpy = self.transform(pet_img_numpy)
#
#         # 缩放 MRI 和 PET 数据
#         if mri_img_numpy.ndim == 3:
#             mri_img_numpy = skt.resize(mri_img_numpy, (181, 217, 181))
#         elif mri_img_numpy.ndim == 4:
#             mri_img_numpy = mri_img_numpy[:, :, :, -1]
#             mri_img_numpy = skt.resize(mri_img_numpy, (181, 217, 181))
#
#         if pet_img_numpy.ndim == 3:
#             pet_img_numpy = skt.resize(pet_img_numpy, (181, 217, 181))
#         elif pet_img_numpy.ndim == 4:
#             pet_img_numpy = pet_img_numpy[:, :, :, -1]
#             pet_img_numpy = skt.resize(pet_img_numpy, (181, 217, 181))
#
#         # 转换为 PyTorch tensors
#         mri_img_torch = torch.from_numpy(mri_img_numpy).unsqueeze(0)
#         pet_img_torch = torch.from_numpy(pet_img_numpy).unsqueeze(0)
#
#         # 将NaN的数据变为0
#         mri_img_torch = torch.where(torch.isnan(mri_img_torch), torch.full_like(mri_img_torch, 0), mri_img_torch)
#         pet_img_torch = torch.where(torch.isnan(pet_img_torch), torch.full_like(pet_img_torch, 0), pet_img_torch)
#
#         # 随机数据增强
#         idx = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()
#         idy = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()
#         idz = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()
#
#         mri_img_torch = mri_img_torch[:, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz]
#         pet_img_torch = pet_img_torch[:, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz]
#
#         # 归一化
#         mri_img_torch = torch.minimum(torch.tensor(1.0, dtype=torch.float32), mri_img_torch.float() / 96.0 - 1.0)
#         pet_img_torch = torch.minimum(torch.tensor(1.0, dtype=torch.float32), pet_img_torch.float() / 96.0 - 1.0)
#
#         label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
#
#         return mri_img_torch, pet_img_torch, label
#
#
# # 创建 DataLoader
# def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#     return dataloader
#
#
# if __name__ == '__main__':
#     # 示例：使用 DataLoader 读取数据
#     mri_dir = Path('/home/publicdata/ADNI/MRI')  # 替换为 MRI 文件的路径
#     pet_dir = Path('/home/publicdata/ADNI/ADNI1_PET_final')  # 替换为 PET 文件的路径
#     csv_file = Path('./Matched_ADNI1.csv')  # 替换为 CSV 文件路径
#     batch_size = 8  # 设置批次大小
#
#     dataset = MriPetDataset(mri_dir, pet_dir, csv_file, valid_group=("AD", "CN"))
#     dataloader = get_dataloader(dataset, batch_size)
#
#     # 测试读取数据
#     print('dataloader', len(dataloader))
#     print('dataset', len(dataloader.dataset))
#     for i, (mri_imgs, pet_imgs, labels) in enumerate(dataloader):
#         print(f"{i} MRI Images batch shape: {mri_imgs.shape}")
#         print(f"{i} PET Images batch shape: {pet_imgs.shape}")
#         print(f"{i} Labels batch shape: {labels}")


import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import transform as skt


# 自定义 Dataset 类
class NiiDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, argument_side=3, valid_group=("AD", "CN")):
        """
        Args:
            data_dir (string or Path): nii.gz 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第二列是文件名，第三列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.data_dir = Path(data_dir)  # 使用 Path 来处理路径
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.transform = transform
        self.argument_side = argument_side
        self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 1, 'sSCD': 0, 'pSCD': 1,
                       'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0, 'sCN': 0,
                       'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.valid_group = valid_group

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]
        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        filtered_idx = self.filtered_indices[idx]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # 获取对应的文件名和标签
        img_name = self.data_dir / (img_name + '.nii')  # 使用 / 操作符拼接路径

        # 使用 nibabel 读取 .nii.gz 文件
        img_numpy = nib.load(str(img_name)).get_fdata()

        # 如果需要，可以对数据进行转换（例如归一化、数据增强等）
        if self.transform:
            img_numpy = self.transform(img_numpy)

        if img_numpy.ndim == 3:
            # 将256等其他尺寸，等比例缩放至168
            img_numpy = skt.resize(img_numpy, (181, 217, 181))
        elif img_numpy.ndim == 4:
            img_numpy = img_numpy[:, :, :, -1]
            img_numpy = skt.resize(img_numpy, (181, 217, 181))

        # 将数据转换为 PyTorch tensor
        img_torch = torch.from_numpy(img_numpy)
        img_torch = img_torch.unsqueeze(0)
        # 将NaN的数据变为0
        img_data = torch.where(torch.isnan(img_torch), torch.full_like(img_torch, 0), img_torch)

        # 随机数据增强
        idx = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()
        idy = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()
        idz = torch.randint(-self.argument_side, self.argument_side + 1, (1,)).item()

        img_torch = img_torch[:, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz]
        img_torch = torch.minimum(torch.tensor(1.0, dtype=torch.float32), img_torch.float() / 96.0 - 1.0)

        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        return img_torch, label


# 创建 DataLoader
def create_custom3d_dataset(dataset, batch_size, transform=None, shuffle=True, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    # 示例：使用 DataLoader 读取数据
    data_dir = Path('/home/publicdata/ADNI/MRI')  # 替换为 .nii.gz 文件所在的路径
    csv_file = Path('/home/shenxiangyuhd/JSRL/test_pytorch/Matched_ADNI1.csv')  # 替换为 CSV 文件路径
    batch_size = 8  # 设置批次大小
    # device = torch.device('cpu')

    dataset = NiiDataset(data_dir, csv_file, valid_group=("AD", "CN"))
    dataloader = create_custom3d_dataset(dataset, batch_size)

    # 测试读取数据
    print('dataloader', len(dataloader))
    print('dataset', len(dataloader.dataset))
    for i, (imgs, labels) in enumerate(dataloader):
        print(f"Images batch shape: {imgs.shape}")
        print(f"Labels batch shape: {labels.shape}")