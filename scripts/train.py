import argparse
import json
from datetime import datetime
import torchvision
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision import tv_tensors
from torchvision.transforms import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.handlers.tensorboard_logger import *

from semantic_segmentation_ros.dataset import SegDataset
from semantic_segmentation_ros.metrics import MeanIoU
from semantic_segmentation_ros.networks import get_model
###全部学習のコードで使っている
def main(config: dict) -> None:
    """
    Main function that handles setup to training, validation, and model saving.

    Inputs:
        config (dict): Configuration settings loaded from a config file.
    Outputs:
        None (The model is trained and results are logged.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create log directory
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    description = f'{time_stamp},model={config["arch"]["model_name"]},batch_size={config["dataloader"]["batch_size"]},lr={config["train"]["lr"]}/'
    logdir = config["log"]["path"] + description 

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(**config["dataset"], **config["dataloader"])

    # Build the network
    model = get_model(**config["arch"]).to(device)

    # Define optimizer, criterion and metrics
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "loss": Loss(criterion),
        "mIOU": MeanIoU(num_classes=config["arch"]["classes"], device=device)
    }

    # Create ignite engines for training and validation
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Log training progress to the terminal and tensorboard
    ProgressBar(persist=False).attach(trainer)
    train_writer, val_writer = create_summary_writers(logdir)

    

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine: Engine) -> None:
        train_evaluator.run(train_loader)
        epoch, metrics = trainer.state.epoch, train_evaluator.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("mIOU", metrics["mIOU"], epoch)

        ###################可視化 これ使うとgpuメモリ不足になるかも
        # model.eval()
        # batch = next(iter(train_loader))  # trainの一部だけ
        # imgs, masks = batch
        # imgs, masks = imgs.to(device), masks.to(device)
        # with torch.no_grad():
        #     preds = model(imgs).argmax(1)
        # visualize_batch_tb(train_writer, imgs, masks, preds, trainer.state.epoch, tag="train/sample")
        # model.train()
        ######################
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine: Engine) -> None:
        val_evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, val_evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("mIOU", metrics["mIOU"], epoch)

        ###############可視化
        # model.eval()
        # batch = next(iter(train_loader))  # trainの一部だけ
        # imgs, masks = batch
        # imgs, masks = imgs.to(device), masks.to(device)
        # with torch.no_grad():
        #     preds = model(imgs).argmax(1)
        # visualize_batch_tb(train_writer, imgs, masks, preds, trainer.state.epoch, tag="val/sample")
        # model.train()
        ##################
    # Checkpoint best model
    #miou大きいのをモデル保存
    model_checkpoint = ModelCheckpoint(
        dirname=str(logdir),
        score_function=lambda engine: engine.state.metrics['mIOU'],
        score_name="mIOU",
        n_saved=15,
        create_dir=True,
        global_step_transform=global_step_from_engine(trainer), 
    )
    ###損失関数小さい方を保存
#     model_checkpoint = ModelCheckpoint(
#         dirname=str(logdir),
#         score_function=lambda engine: -engine.state.metrics['loss'],  # 損失を小さい順に保存
#         score_name="loss",
#         n_saved=15,
#         create_dir=True,
#         global_step_transform=global_step_from_engine(trainer), 
# )

    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {config["arch"]["model_name"]: model})
    # Run the training loop
    trainer.run(train_loader, max_epochs=config["train"]["epochs"])

def create_train_val_loaders(path: str, labels: list, augmentations: dict, batch_size: int, num_workers: int, pin_memory: bool) -> tuple:
    """
    Generate loaders for training and validation datasets.

    Inputs:
        path (str): Path to the image data.
        labels (list): List of labels.
        augmentations (dict): Augmentations to be applied as preprocessing.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loader.
        pin_memory (bool): Whether to use pin memory.

    Outputs:
        train_loader (DataLoader), val_loader (DataLoader): Loaders for training and validation datasets.
    """
    train_dataset = SegDataset(path + "/train", labels, augmentations, is_train=True)
    val_dataset = SegDataset(path + "/val", labels, augmentations, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def create_summary_writers(log_dir: str) -> tuple:
    """
    Create TensorBoard summary writers for logging.

    Inputs:
        log_dir (str): Path to the log directory.
    Outputs:
        Tuple containing the training and validation SummaryWriters.
    """
    train_path = log_dir + "train"
    val_path = log_dir + "validation"
    
    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)
    return train_writer, val_writer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/train.json', help="Path to config JSON file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    return config








#################################

def visualize_batch_tb(writer, images, targets, preds, epoch, tag="sample"):
    """
    TensorBoardに画像・GTマスク・予測マスクを可視化

    images : (B, C, H, W)
    targets: (B, H, W) or (B, C, H, W)
    preds  : (B, H, W) or (B, C, H, W)
    """
    images = images.cpu()
    targets = targets.cpu()
    preds = preds.cpu()
    
    # print("images:", images.shape)
    # print("targets:", targets.shape)
    # print("preds:", preds.shape)

    # 常に1枚だけ表示
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # 画像 (B,C,H,W) → (H,W,C)
    img = images[0].permute(1, 2, 0).numpy()

    # GT (B,H,W) or (B,1,H,W) → (H,W)
    gt = targets[0]
    if gt.ndim == 3:  # one-hotとか余分な次元がある場合
        gt = torch.argmax(gt, dim=0)
    gt = gt.numpy()

    # Pred (B,H,W) or (B,C,H,W) → (H,W)
    pred = preds[0]
    
    if pred.ndim == 3:  # (C,H,W)
        pred = torch.argmax(pred, dim=0)
    pred = pred.numpy()

    axes[0].imshow(img.astype("uint8"))
    axes[0].set_title("Input"); axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT Mask"); axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Pred Mask"); axes[2].axis("off")

    plt.tight_layout()
    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)





if __name__ == "__main__":
    config = parse_args()
    main(config)