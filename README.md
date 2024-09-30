# 3D_Diffusion_MRI_PET

## 1、训练阶段：

写好`config.yml`文件，然后运行

```
python3 train.py
```

## 2、采样阶段

1、只保存最后的采样结果

```bash
python generate.py -cp "checkpoint/cifar10.pth" -bs 16 -sp "./cifar10_result.png" --nrow 4 --result_only --weight=1.8

```

1、保存中间的采样过程

```bash
python generate.py -cp "checkpoint/cifar10.pth" -bs 16 -sp "result/cifar10_sampler.png" --weight=1.8
```

