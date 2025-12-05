import os
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
from network import CustomDigitCNN, CustomDigitCNNRes
from config import cfg, EasyConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

import torchvision
from torchvision import datasets, transforms, models

from utils import mser
from torch.utils.tensorboard import SummaryWriter

import random

import time
import re
MODE = "train"
DATA_ROOT = "data"
CHECKPOINT_DIR = "checkpoints"
GRADED_OUTPUT_DIR = "graded_images"
DEMO_IMAGE_DIR = "test"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GRADED_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEMO_IMAGE_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
   
NUM_CLASSES = 10


class ToGray(object):
    def __call__(self, t):
        if t.shape[0] == 3:
            r, g, b = t[0], t[1], t[2]
            gray_t = 0.299 * r + 0.587 * g + 0.114 * b
            return gray_t.unsqueeze(0)
        return t

class AddGaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, t):
        noise = torch.randn_like(t) * self.std
        t = t + noise
        t = torch.clamp(t, 0.0, 1.0)
        return t

def get_svhn_transforms(gray: bool = True, noise_aug: bool = True, train: bool = True):

    tfm_list = []
    if train:
        tfm_list.extend([
            transforms.Resize((40, 40)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.CenterCrop((32, 32)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])
    else:
        tfm_list.append(transforms.Resize((32, 32)))

    tfm_list.append(transforms.ToTensor())


    if gray:
        tfm_list.append(ToGray())   # output shape: (1,H,W)

        tfm_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        tfm_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5]))

    if noise_aug and train:
        tfm_list.append(AddGaussianNoise(0.05))

    return transforms.Compose(tfm_list)



def build_vgg16_digit_model(num_classes, freeze_features=False):
    vgg = models.vgg16(pretrained=False)
    vgg.load_state_dict(torch.load("weight/vgg16-397923af.pth"))
    if freeze_features:
        for p in vgg.features.parameters():
            p.requires_grad = False
    in_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(in_features, num_classes)
    return vgg

def debug_check_labels(train_loader):
    imgs, labels = next(iter(train_loader))
    print("Label min:", labels.min().item())
    print("Label max:", labels.max().item())
    print("Unique labels:", torch.unique(labels))

def build_model(cfg: EasyConfig) -> nn.Module:
    model_type = cfg.MODEL.TYPE
    if model_type == "CustomCNN":
        model = CustomDigitCNNRes(num_classes=NUM_CLASSES)
    elif model_type == "vgg16":
        model = build_vgg16_digit_model(
            num_classes=NUM_CLASSES,
            freeze_features=cfg.MODEL.FREEZE_FEATURES
        )
    else:
        raise ValueError(f"Unknown MODEL.TYPE: {model_type}")
    return model

def model_fn(model, batch, device):
    """
    model_fn_decorator
    batch: (inputs, labels)
    return:
        loss, tb_dict, disp_dict
    """
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    _, preds = outputs.max(1)
    # print(preds)
    # import pdb; pdb.set_trace()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total

    tb_dict = {
        "loss": loss.item(),
        "acc": acc,
    }
    disp_dict = {
        "loss": f"{loss.item():.4f}",
        "acc": f"{acc:.4f}",
    }
    return loss, tb_dict, disp_dict


def build_optimizer(cfg: EasyConfig, model: nn.Module):
    lr = cfg.TRAIN.LR
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

def build_dataloader(cfg: EasyConfig):
    batch_size = cfg.DATA.BATCH_SIZE
    gray = cfg.DATA.GRAY

    train_transform = get_svhn_transforms(gray=gray, noise_aug=False, train=False)
    test_transform = get_svhn_transforms(gray=gray, noise_aug=False, train=False)

    train_dataset = datasets.SVHN(
        root=DATA_ROOT,
        split='train',
        transform=train_transform,
        download=True
    )

    # for debuggin
    # small_n = 512
    # subset_indices = list(range(small_n))
    # train_dataset = Subset(train_dataset, subset_indices)

    test_dataset = datasets.SVHN(
        root=DATA_ROOT,
        split='test',
        transform=test_transform,
        download=True
    )

    val_ratio = cfg.DATA.VAL_SPLIT
    val_size = int(len(test_dataset) * val_ratio)
    test_size = len(test_dataset) - val_size
    test_ds, val_ds = random_split(test_dataset, [test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    debug_check_labels(train_loader)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


class Trainer:
    def __init__(self, cfg: EasyConfig, resume=True):
        self.cfg = cfg
        self.device = DEVICE
        print(f"[Device] Training on device: {self.device}")
        # dataloader / model / optimizer / scheduler
        self.train_loader, self.val_loader, self.test_loader = build_dataloader(cfg)
        self.model = build_model(cfg).to(self.device)
        self.optimizer, self.scheduler = build_optimizer(cfg, self.model)
        print(f"[Device] Training on device: {self.device}")  
        self.tb_writer = SummaryWriter(log_dir="tb_logs")

        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.ckpt_path = os.path.join(CHECKPOINT_DIR, cfg.TRAIN.CKPT_NAME)

        if resume:
            self._try_resume()
        self.debug_single_batch_step()

    def debug_single_batch_step(self):
        self.model.train()
        batch = next(iter(self.train_loader))
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        print("Initial labels:", labels[:16])

        criterion = nn.CrossEntropyLoss()
        outputs = self.model(inputs)
        loss_before = criterion(outputs, labels).item()
        print(f"[Debug] loss before step: {loss_before:.4f}")

        self.optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        outputs_after = self.model(inputs)
        loss_after = criterion(outputs_after, labels).item()
        print(f"[Debug] loss after  step: {loss_after:.4f}")

    def _try_resume(self):

        base_name = self.cfg.TRAIN.CKPT_NAME.replace("_best.pth", "")
        pattern = re.compile(rf"{base_name}_epoch_(\d+)\.pth")

        candidates = []
        for fname in os.listdir(CHECKPOINT_DIR):
            match = pattern.match(fname)
            if match:
                epoch_num = int(match.group(1))
                candidates.append((epoch_num, fname))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            last_epoch, last_file = candidates[-1]
            resume_path = os.path.join(CHECKPOINT_DIR, last_file)
            print(f"[Resume] Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device)

            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

            self.start_epoch = ckpt["epoch"]
            self.best_val_acc = ckpt.get("val_acc", 0.0)

            print(f"[Resume] Resumed from epoch {self.start_epoch} (best val_acc={self.best_val_acc:.4f})")
        else:
            print(f"[Resume] No checkpoint found at {CHECKPOINT_DIR}, starting fresh.")

    def save_checkpoint(self, epoch, val_acc, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_acc": val_acc,
            "cfg": dict(self.cfg),
        }

        per_epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"{self.cfg.TRAIN.CKPT_NAME.replace('_best.pth','')}_epoch_{epoch}.pth"
        )
        torch.save(state, per_epoch_path)
        if is_best:
            torch.save(state, self.ckpt_path)
            print(f"  [CKPT] New best val_acc={val_acc:.4f} saved at {self.ckpt_path}")

    def train_one_epoch(self, epoch):

        self.model.train()
        log_interval = self.cfg.TRAIN.LOG_INTERVAL

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for it, batch in enumerate(self.train_loader):

            global_iter = (epoch - 1) * len(self.train_loader) + it
            loss, tb_dict, disp_dict = model_fn(self.model, batch, self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # accumulate stats
            bs = batch[1].size(0)
            total_loss += tb_dict["loss"] * bs
            total_correct += tb_dict["acc"] * bs
            total_samples += bs
            # TensorBoard iteration-l
            # evel logging
            self.tb_writer.add_scalar("Train/iter_loss", tb_dict["loss"], global_iter)
            self.tb_writer.add_scalar("Train/iter_acc", tb_dict["acc"], global_iter)


            if (it + 1) % log_interval == 0:
                print(f"  [Train][Epoch {epoch}][Iter {it+1}/{len(self.train_loader)}] "
                      f"loss={disp_dict['loss']} acc={disp_dict['acc']}")

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        # TensorBoard epoch-level
        self.tb_writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)
        self.tb_writer.add_scalar("Train/epoch_acc", epoch_acc, epoch)
        self.tb_writer.add_scalar("Train/lr", self.scheduler.get_last_lr()[0], epoch)
        print(f"[Train][Epoch {epoch}] loss={epoch_loss:.4f} acc={epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def eval_one_epoch(self, epoch, mode="val"):
        self.model.eval()
        loader = self.val_loader if mode == "val" else self.test_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            loss, tb_dict, disp_dict = model_fn(self.model, batch, self.device)

            bs = batch[1].size(0)
            total_loss += tb_dict["loss"] * bs
            total_correct += tb_dict["acc"] * bs
            total_samples += bs

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples

        # TensorBoard logging
        tag = "Val" if mode == "val" else "Test"
        self.tb_writer.add_scalar(f"{tag}/loss", epoch_loss, epoch)
        self.tb_writer.add_scalar(f"{tag}/acc", epoch_acc, epoch)

        print(f"[{mode.capitalize()}][Epoch {epoch}] loss={epoch_loss:.4f} acc={epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    def train(self):
        num_epochs = self.cfg.TRAIN.NUM_EPOCHS

        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            iter_start = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            iter_end = time.time()
            iter_time = iter_end - iter_start
            print(f"time={iter_time:.4f}s")
            val_loss, val_acc = self.eval_one_epoch(epoch, mode="val")

            # scheduler step
            self.scheduler.step()

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, val_acc, is_best)

        print("==> Evaluating best checkpoint on test set")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc = self.eval_one_epoch(ckpt["epoch"], mode="test")
        print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

        self.tb_writer.close()
def read_sequence(
    boxes,
    predictions,
):
    valid = []
    for (x, y, w, h), (digit, prob) in zip(boxes, predictions):
        if digit is not None:
            valid.append((x, digit, prob))

    valid.sort(key=lambda x: x[0])
    seq = "".join(str(v[1]) for v in valid)
    return seq

def classify_digits(
        model,
        patches,
        thresh,
        device,
        use_gray,
):
    if use_gray:
        tfm = get_svhn_transforms(gray=True, noise_aug=False, train=False)
    else:
        tfm = get_svhn_transforms(gray=False, noise_aug=False, train=False)

    # preprocessing: resize image
    inputs = []
    for patch in patches:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        t = tfm(pil)
        inputs.append(t)

    batch = torch.stack(inputs, dim=0).to(device)
    outputs = model(batch)
    probs = torch.softmax(outputs, dim=1)
    conf, preds = probs.max(dim=1)

    results =[]
    for c, p in zip(conf.detach().cpu().numpy(), preds.detach().cpu().numpy()):
        if c < thresh: # background
            results.append((None, float(c)))
        else:
            results.append((int(p), float(c)))

    return results

def visualize(image_bgr, boxes, preds):
    # Visualization
    vis = image_bgr.copy()
    for (x, y, w, h), (digit, prob) in zip(boxes, preds):
        if digit is None:
            continue
        text = f"{digit}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return vis

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def demo(
        cfg
):
    
    if cfg.MODEL.TYPE =="vgg16":
        model = build_vgg16_digit_model(num_classes=NUM_CLASSES, freeze_features=False)
        ckpt_path = os.path.join(CHECKPOINT_DIR, cfg.TRAIN.CKPT_NAME)
        use_gray = False 
    else:
        model = CustomDigitCNNRes(num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(CHECKPOINT_DIR, cfg.TRAIN.CKPT_NAME) 
        use_gray=True

    model = model.to(DEVICE)
    count_params(model)

    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    thresh = cfg.DEMO.THRESH

    all_images = [
        f for f in os.listdir(DEMO_IMAGE_DIR)
        if f.lower().endswith(".png")
    ]
    # import pdb; pdb.set_trace()
    sampled = random.sample(all_images, 100)
    for i in range(len(sampled)):
        # in_path = sampled[i]
        in_path = os.path.join(DEMO_IMAGE_DIR, sampled[i])
        out_path = os.path.join(GRADED_OUTPUT_DIR, sampled[i])

        image_bgr = cv2.imread(in_path)
        boxes = mser(image_bgr)
        rois = []
        for (x, y, w, h) in boxes:
            roi = image_bgr[y:y+h, x:x+w]
            rois.append(roi)
        preds = classify_digits(model, rois,thresh, DEVICE, use_gray=use_gray )
        seq = read_sequence(boxes, preds)

        vis=visualize(image_bgr, boxes, preds)

        if out_path is not None:
            cv2.imwrite(out_path, vis)

        print(f"[{os.path.basename(in_path)}] predicted sequence: {seq}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--model_type", type=str, required=True, choices=["CustomCNN", "vgg16"])
    return parser.parse_args()

def main():
    args = parse_args()
    cfg.MODEL.TYPE = args.model_type

    if cfg.MODEL.TYPE == "vgg16":
        cfg.DATA.GRAY = False
        cfg.TRAIN.CKPT_NAME = "vgg16_best.pth"
    else:
        cfg.DATA.GRAY = True
        cfg.TRAIN.CKPT_NAME = "custom_cnn_res_best.pth"

    print("******* Config SETTING *******")
    print(cfg)
    print("******* Config SETTING *******")

    if args.mode == "train":
        trainer = Trainer(cfg)
        trainer.train()
    elif args.mode == "test":
        demo(cfg)

if __name__ == "__main__":
    main()
