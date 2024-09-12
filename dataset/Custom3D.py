import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path

# 1. 读取 .nii.gz 数据
def load_nii_data(file_path, augment = True, argument_side = 3):
    img = nib.load(file_path)
    img_data_numpy = img.get_fdata()
    img_data = torch.from_numpy(img_data_numpy).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    # 将NaN的数据变为0
    img_data = torch.where(torch.isnan(img_data), torch.full_like(img_data, 0), img_data)
    if augment:
        idx = torch.randint(argument_side, argument_side + 1, (1,)).item()
        idy = torch.randint(argument_side, argument_side + 1, (1,)).item()
        idz = torch.randint(argument_side, argument_side + 1, (1,)).item()
    else:
        idx = 0
        idy = 0
        idz = 0

    # 调整输出尺寸为与输入 x 相同
    # nii_data = F.interpolate(nii_data, size=[176, 208, 176], mode='trilinear', align_corners=False)

    img_data = img_data[:, :, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz]
    img_data = torch.minimum(torch.tensor(1.0, dtype=torch.float32), img_data.float() / 96.0 - 1.0)

    # return img_data / np.max(img_data)  # 归一化到 [0, 1]
    return img_data


# 2. 批量读取文件夹下所有 .nii.gz 文件
def load_dataset(data_dir, device, augment = True, argument_side = 3):
    data_list = []
    # nii_files = Path(data_dir).glob('*.nii.gz')
    nii_files = []
    # 分别匹配 .nii 和 .nii.gz 文件
    nii_files.extend(Path(data_dir).glob('*.nii'))
    nii_files.extend(Path(data_dir).glob('*.nii.gz'))
    for nii_file in nii_files:
            nii_data = load_nii_data(nii_file, augment, argument_side).to(device)
            data_list.append(nii_data)
    return data_list

# 2. 批量读取文件夹下所有 .nii.gz 文件，将读取的数据写入文件
# def load_dataset(data_dir, device, output_file='output.txt', augment=True, argument_side=3):
#     data_list = []
#     nii_files = []
#     # 匹配 .nii 和 .nii.gz 文件
#     nii_files.extend(Path(data_dir).glob('*.nii'))
#     nii_files.extend(Path(data_dir).glob('*.nii.gz'))
#     with open(output_file, 'w') as f:  # 打开文件写入模式
#         for nii_file in nii_files:
#             nii_data = load_nii_data(nii_file, augment, argument_side).to(device)
#             data_list.append(nii_data)
#             # 获取文件名和shape信息，写入到文件中
#             f.write(f"File: {nii_file.name}, Shape: {list(nii_data.shape)}\n")
#
#     return data_list

# if __name__ == '__main__':
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     mri_datalist = load_dataset('', device)
#     for data in mri_datalist:
#         print(data.shape)