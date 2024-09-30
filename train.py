from dataset import create_dataset
from model.UNet import UNet
from model.UnetCondition import UNetCondition
from model.UNet3D import UNet3D
from utils.engine import GaussianDiffusionTrainer, GaussianDiffusionConditionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch
from utils.callbacks import ModelCheckpoint


def train(config):
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)

    device = torch.device(config["device"])
    class_num, loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    # model = UNet3D(**config["Model"]).to(device)
    if config['use_label']:
        model = UNetCondition(num_labels=class_num, **config["Model"]).to(device)
        trainer = GaussianDiffusionConditionTrainer(model, **config["Trainer"]).to(device)
    else:
        model = UNet(**config["Model"]).to(device)
        trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    print(f"参数量：{sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, config['use_label'])
        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
