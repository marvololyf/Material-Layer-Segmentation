import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 数据集定位到根目录下的 data/images 和 data/masks
ORIGINAL_IMG = os.path.join(PROJECT_ROOT, "data", "images")
ORIGINAL_MASK = os.path.join(PROJECT_ROOT, "data", "masks")
IMAGE_DIR, MASK_DIR = ORIGINAL_IMG, ORIGINAL_MASK

# 在 train_yuanben.py 开头加入配置读取
def load_config():
    possible_config_paths = [
        os.path.join(PROJECT_ROOT, "configs", "config.json"),
        os.path.join(SCRIPT_DIR, "config.json"),
    ]
    for path in possible_config_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

CONFIG = load_config()

IMAGE_SIZE = tuple(CONFIG.get("settings", {}).get("image_size", [768, 768]))
NUM_CLASSES = CONFIG.get("settings", {}).get("num_classes", 4)


raw_model_path = CONFIG.get("paths", {}).get("model_path", "models/best_unet_model_v3.pth")

if os.path.isabs(raw_model_path):
    MODEL_SAVE_PATH = raw_model_path
else:
    # 基于根目录 PROJECT_ROOT 拼出保存路径: Material-Layer-Segmentation/models/best_unet_model_v3.pth
    MODEL_SAVE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_model_path))

# 自动创建 models/ 目标文件夹（避免首次训练保存权重时报 FileNotFoundError）
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 80
PATIENCE = 20


CLASS_NAMES = {
    0: "树脂层",
    1: "陶瓷层",
    2: "粘结层",
    3: "基体层"
}


# ================= 1. 基础 DoubleConv 模块 =================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ================= 2. ResNet34-UNet =================
class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=4, freeze_front_layers=True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 冻结前部层
        if freeze_front_layers:
            for module in [self.stem, self.layer1, self.layer2]:
                for param in module.parameters():
                    param.requires_grad = False
            print("🔒 已成功冻结 ResNet34 的前部层 (Stem + Layer1 + Layer2)")

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_dec = DoubleConv(32, 32)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)                    # 1/2
        x1 = self.layer1(self.maxpool(x0))  # 1/4
        x2 = self.layer2(x1)                # 1/8
        x3 = self.layer3(x2)                # 1/16
        x4 = self.layer4(x3)                # 1/32

        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))

        out = self.final_dec(self.final_up(d1))
        return self.out_conv(out)


# ================= 3. 数据增强与数据集 =================
class PureDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])

        if is_train:
            self.transform = A.Compose([
                A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.8),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def _load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"未找到 mask 文件: {mask_path}")

        unique_vals = np.unique(mask)

        # 理想情况：已经是 0/1/2/3
        if np.all(np.isin(unique_vals, [0, 1, 2, 3])):
            return mask

        # 兼容性处理：若仍存在旧灰度编码，则按排序重映射到 0..3
        print(f"⚠️ 警告: mask {mask_path} 存在非 0/1/2/3 标签值 {unique_vals.tolist()}，将按排序重映射。")
        sorted_vals = sorted(unique_vals.tolist())
        mapping = {}
        for i, val in enumerate(sorted_vals[:NUM_CLASSES]):
            mapping[val] = i

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for old_val, new_val in mapping.items():
            new_mask[mask == old_val] = new_val

        return new_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        stem_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, stem_name + ".png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"未找到图像文件: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self._load_mask(mask_path)

        transformed = self.transform(image=image, mask=mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].long()

        return image_t, mask_t


# ================= 4. mIoU MetricTracker（4类都参与） =================
class MetricTracker:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()

        mask = (targets >= 0) & (targets < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets[mask].astype(int) + preds[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist

    def get_miou(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        iou = np.where(union != 0, intersection / union, np.nan)
        mean_iou = np.nanmean(iou)

        return mean_iou, iou


# ================= 5. Dice Loss =================
class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, weights=None, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score

        if self.weights is not None:
            dice_loss = dice_loss * self.weights
            return dice_loss.mean()

        return dice_loss.mean()


# ================= 6. 联合损失函数 =================
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 0=树脂层, 1=陶瓷层, 2=粘结层, 3=基体层
        # 现在 0 也是真实类别，不再当背景压权
        weights = torch.tensor([1.0, 1.0, 2.5, 1.0], dtype=torch.float32).to(DEVICE)

        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = MulticlassDiceLoss(num_classes=NUM_CLASSES, weights=weights)

    def forward(self, y_pred, y_true):
        ce_loss = self.ce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return 0.4 * ce_loss + 0.6 * dice_loss


# ================= 7. 训练主流程 =================
def train_model():
    train_img_dir = os.path.join(IMAGE_DIR, "train")
    train_mask_dir = os.path.join(MASK_DIR, "train")
    val_img_dir = os.path.join(IMAGE_DIR, "val")
    val_mask_dir = os.path.join(MASK_DIR, "val")

    train_dataset = PureDataset(train_img_dir, train_mask_dir, is_train=True)
    val_dataset = PureDataset(val_img_dir, val_mask_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ResNet34UNet(
        num_classes=NUM_CLASSES,
        freeze_front_layers=True
    ).to(DEVICE)

    criterion = SegLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        trainable_params,
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5
    )

    metric_tracker = MetricTracker(num_classes=NUM_CLASSES)
    best_val_miou = 0.0
    no_improve_count = 0

    print(f"🚀 开始训练 (4类分割 | 设备: {DEVICE})")
    print(f"📁 IMAGE_DIR: {IMAGE_DIR}")
    print(f"📁 MASK_DIR : {MASK_DIR}")
    print(f"🏷️ 类别定义: 0=树脂层, 1=陶瓷层, 2=粘结层, 3=基体层")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS} [训练]"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        metric_tracker.reset()

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS} [验证]"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                metric_tracker.update(preds, masks)

        val_loss /= max(1, len(val_loader))
        mean_iou, class_ious = metric_tracker.get_miou()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"📊 Epoch {epoch + 1:02d} => "
            f"训练Loss: {train_loss:.4f}, "
            f"验证Loss: {val_loss:.4f}, "
            f"4类mIoU: {mean_iou * 100:.2f}%, "
            f"LR: {current_lr:.2e}"
        )

        print(
            "   ↳ 各类IoU "
            f"[0:树脂层: {class_ious[0] * 100:.2f}%, "
            f"1:陶瓷层: {class_ious[1] * 100:.2f}%, "
            f"2:粘结层: {class_ious[2] * 100:.2f}%, "
            f"3:基体层: {class_ious[3] * 100:.2f}%]"
        )

        if mean_iou > best_val_miou:
            best_val_miou = mean_iou
            no_improve_count = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 4类 mIoU 刷新记录 ({mean_iou * 100:.2f}%)！模型已保存至:\n   {MODEL_SAVE_PATH}\n")
        else:
            no_improve_count += 1
            print(f"⚠️ 4类 mIoU 已连续 {no_improve_count}/{PATIENCE} 轮未改善\n")

        if no_improve_count >= PATIENCE:
            print(f"🛑 连续 {PATIENCE} 轮未改善，触发提前停止 (Early Stopping)！")
            break

    print(f"🎉 训练完成，最高 4类 mIoU: {best_val_miou * 100:.2f}%")


if __name__ == "__main__":
    train_model()