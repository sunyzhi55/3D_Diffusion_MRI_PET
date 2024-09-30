from utils.engine import DDPMSampler, DDIMSampler, DDPMConditionSampler
from model.UNet import UNet
from model.UnetCondition import UNetCondition
from model.UNet3D import UNet3D
from dataset import create_dataset
import torch
from utils.tools import save_sample_image, save_image
from argparse import ArgumentParser
from pathlib import Path

def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # Condition sampler param
    parser.add_argument("--weight", type=float, default=1.8)

    # DDIM sampler param
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)
    max_class_num, _ = create_dataset(**cp["config"]["Dataset"])
    assert max_class_num >= args.nrow, "nrow must be smaller than max_class_num"


    # load trained model
    if cp["config"]['use_label']:
        model = UNetCondition(num_labels=max_class_num, **cp["config"]["Model"])
    else:
        model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model = model.eval()
    # generate Gaussian noise
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
                       *cp["config"]["Dataset"]["image_size"]), device=device)

    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)

    # define the sampler
    if cp["config"]['use_label']:
        sampler = DDPMConditionSampler(model,**cp["config"]["Trainer"], w=args.weight).to(device)
        # labels = []

        # batch_size = 64
        # class_num = 4
        # max_class = 20  # 可以根据需要调整最大值
        step = args.batch_size // args.nrow
        label_list = []

        for _ in range(args.nrow):
            value = torch.randint(0, max_class_num + 1, (1,)).item()  # 取出数字
            label_list.append(torch.full((step,), value).long())

        labels = torch.cat(label_list, dim=0).to(device)

        x = sampler(z_t, labels, only_return_x_0=args.result_only, interval=args.interval, **extra_param)
    else:

        if args.sampler == "ddim":
            sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
        elif args.sampler == "ddpm":
            sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
        else:
            raise ValueError(f"Unknown sampler: {args.sampler}")
        x = sampler(z_t, only_return_x_0=args.result_only, interval=args.interval, **extra_param)

    # 保存图片路径

    Path(args.image_save_path).parent.mkdir(parents=True, exist_ok=True)
    if args.result_only:
        save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    else:
        save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)


if __name__ == "__main__":
    args = parse_option()
    generate(args)
'''
python generate.py -cp "checkpoint/cifar10.pth" -bs 16 -sp "./cifar10_result.png" --nrow 4 --result_only --weight=1.8
python generate.py -cp "checkpoint/cifar10.pth" -bs 16 -sp "data/result/cifar10_sampler.png" --weight=1.8
'''