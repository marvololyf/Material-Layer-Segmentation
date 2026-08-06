import os
import re
import json
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageTk


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def load_config():
    possible_config_paths = [
        os.path.join(PROJECT_ROOT, "configs", "config.json"),
        os.path.join(SCRIPT_DIR, "config.json"),
    ]
    
    config_path = None
    for path in possible_config_paths:
        if os.path.exists(path):
            config_path = path
            break

    if not config_path:
        raise FileNotFoundError(
            f"找不到配置文件 config.json，尝试查找路径:\n" + 
            "\n".join(possible_config_paths)
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGORITHM_VERSION = "2026-07-20-resunet34-layer-boundary-v3"
raw_model_path = CONFIG["paths"].get("model_path", "models/best_unet_model_v3.pth")
if os.path.isabs(raw_model_path):
    MODEL_PATH = raw_model_path
else:
    MODEL_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_model_path))

INPUT_WIDTH = CONFIG["settings"]["image_size"][0]
INPUT_HEIGHT = CONFIG["settings"]["image_size"][1]
NUM_CLASSES = CONFIG["settings"].get("num_classes", 4)
THICKNESS_SAMPLE_COUNT = CONFIG["settings"].get("thickness_sample_count", 50)

SCALE_ROI_LEFT_RATIO = CONFIG["settings"].get("scale_roi_left_ratio", 0.7)
SCALE_ROI_TOP_RATIO = CONFIG["settings"].get("scale_roi_top_ratio", 0.8)

# OCR 调试输出目录
OCR_DEBUG_ROOT = os.path.join(PROJECT_ROOT, "ocr_debug")

# 是否保存比例尺 OCR 的裁剪图、预处理图和原始识别结果
OCR_DEBUG_ENABLED = CONFIG["settings"].get(
    "ocr_debug_enabled",
    True,
)

def get_tesseract_cmd():
    config_cmd = CONFIG["paths"].get("tesseract_cmd", "")
    if os.path.isfile(config_cmd):
        return config_cmd
    system_cmd = shutil.which("tesseract")
    if system_cmd:
        return system_cmd
    common_paths = [
        r"C:/Program Files/Tesseract-OCR/tesseract.exe",
        r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path
    return config_cmd


TESSERACT_CMD = get_tesseract_cmd()

CLASS_NAMES = {
    0: "树脂层",
    1: "陶瓷层",
    2: "粘结层",
    3: "基体层",
}

# 高对比度 Mask 叠加颜色 (RGB)
MASK_COLORS = {
    0: (120, 120, 120),  # 树脂层: 暗灰色
    1: (255, 185, 0),    # 陶瓷层: 鲜黄色 (第1层)
    2: (0, 200, 220),    # 粘结层: 鲜青色 (第2层)
    3: (190, 60, 220),   # 基体层: 高亮紫色/品红 (第3层 - 明显醒目)
}

BOUNDARY_NAMES = {
    "resin_ceramic": "树脂/陶瓷分界",
    "ceramic_bonding": "陶瓷/粘结分界",
    "bonding_substrate": "粘结/基体分界",
}

BOUNDARY_COLORS = {
    "resin_ceramic": (255, 40, 40),      # 红色划线
    "ceramic_bonding": (40, 220, 40),    # 绿色划线
    "bonding_substrate": (40, 120, 255),  # 蓝色划线
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONTENT_DARK_THRESHOLD = 18
CONTENT_MIN_AREA_RATIO = 0.20
CONTENT_INNER_MARGIN = 4
EDGE_COLUMN_IGNORE_RATIO = 0.015

MIN_COMPONENT_AREA = 80
MIN_COMPONENT_WIDTH_RATIO = 0.015

BOUNDARY_SMOOTH_RATIO = 0.02
BOUNDARY_MEDIAN_KERNEL = 15
BOUNDARY_GAUSSIAN_KERNEL = 11
BOUNDARY_MIN_COLUMNS_ABS = 16
BOUNDARY_MIN_COLUMNS_RATIO = 0.03
MIN_BOUNDARY_GAP = 4

PORE_MIN_COMPONENT_AREA = 6


def get_mode_class_map(mode):
    if mode == "four_layer":
        return {"resin": 0, "ceramic": 1, "bonding": 2, "substrate": 3}
    else:
        return {"resin": None, "ceramic": 1, "bonding": 2, "substrate": 3}


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


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        resnet = models.resnet34(weights=None)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], dim=1))

        out = self.final_dec(self.final_up(d1))
        return self.out_conv(out)


def imread_rgb(path):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1] or ".png"
    ok, data = cv2.imencode(ext, bgr)
    if not ok:
        raise ValueError(f"无法保存图像：{path}")
    data.tofile(path)


def preprocess_image(image_rgb):
    resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # --- 同步加入 CLAHE，保证预测端与训练端图像特征 100% 对齐 ---
    gray_lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_lab[:, :, 0] = clahe.apply(gray_lab[:, :, 0])
    resized = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    # -------------------------------------------------------------

    image = resized.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0).float()


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型权重：\n{MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model = ResNet34UNet(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_outputs(model, image_rgb):
    original_h, original_w = image_rgb.shape[:2]
    tensor = preprocess_image(image_rgb).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    pred = np.argmax(probs, axis=0).astype(np.uint8)

    pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return pred


def overlay_segmentation_mask(image_rgb, pred_mask, alpha=0.35):
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in MASK_COLORS.items():
        color_mask[pred_mask == class_id] = color

    overlayed = cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask, alpha, 0)
    return overlayed


def ensure_odd(value, minimum=3):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def smooth_1d(values, kernel_size=21):
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return values.copy()
    kernel_size = min(ensure_odd(kernel_size), ensure_odd(values.size))
    if kernel_size > values.size:
        kernel_size = values.size if values.size % 2 == 1 else values.size - 1
    if kernel_size < 3:
        return values.copy()
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def median_filter_1d(values, kernel_size):
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return values.copy()
    kernel_size = min(ensure_odd(kernel_size), values.size)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return values.copy()
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1).astype(np.float32)


def detect_content_roi(image_rgb):
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > CONTENT_DARK_THRESHOLD).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(contour)
    if cw * ch < w * h * CONTENT_MIN_AREA_RATIO:
        return 0, 0, w, h

    x0 = max(0, x + CONTENT_INNER_MARGIN)
    y0 = max(0, y + CONTENT_INNER_MARGIN)
    x1 = min(w, x + cw - CONTENT_INNER_MARGIN)
    y1 = min(h, y + ch - CONTENT_INNER_MARGIN)
    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return 0, 0, w, h
    return x0, y0, x1, y1


def get_scale_exclusion_roi(image_rgb):
    h, w = image_rgb.shape[:2]
    x0 = int(round(w * float(SCALE_ROI_LEFT_RATIO)))
    y0 = int(round(h * float(SCALE_ROI_TOP_RATIO)))
    return max(0, min(w, x0)), max(0, min(h, y0)), w, h


def mask_out_scale_region(pred_mask, image_rgb):
    masked = pred_mask.copy()
    x0, y0, x1, y1 = get_scale_exclusion_roi(image_rgb)
    invalid_class_id = np.uint8(min(255, NUM_CLASSES))
    masked[y0:y1, x0:x1] = invalid_class_id
    return masked


def get_valid_x_range(width, roi=None):
    if roi is None:
        x0, x1 = 0, width
    else:
        x0, _, x1, _ = roi
    margin = max(2, int(width * EDGE_COLUMN_IGNORE_RATIO))
    x0 = min(width - 1, max(0, x0 + margin))
    x1 = max(x0 + 1, min(width, x1 - margin))
    return x0, x1


def filter_small_components(binary_mask):
    h, w = binary_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if count <= 1:
        return binary_mask

    min_width = max(3, int(w * MIN_COMPONENT_WIDTH_RATIO))
    kept = np.zeros((h, w), dtype=np.uint8)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        if area >= MIN_COMPONENT_AREA and width >= min_width:
            kept[labels == label_id] = 1

    if kept.sum() < max(30, binary_mask.sum() * 0.2):
        return binary_mask
    return kept


def largest_component_mask(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if count <= 1:
        return binary_mask * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    result = np.zeros_like(binary_mask, dtype=np.uint8)
    result[labels == best] = 255
    return result


def clean_layer_mask(layer_mask):
    binary = (layer_mask > 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = filter_small_components(binary)
    return (binary * 255).astype(np.uint8)


def curve_from_layer_top(pred_mask, class_id, roi=None):
    if class_id is None:
        return None

    h, w = pred_mask.shape
    
    # 抽取边界前先对二值 Mask 清理形态学噪点
    cleaned_binary = clean_layer_mask(pred_mask == class_id)
    
    x0, x1 = get_valid_x_range(w, roi)
    y0 = 0
    y1 = h
    if roi is not None:
        _, y0, _, y1 = roi

    xs = []
    ys = []
    for x in range(x0, x1):
        col = np.where(cleaned_binary[y0:y1, x] > 0)[0]
        if col.size > 0:
            xs.append(x)
            ys.append(y0 + int(col.min()))

    min_columns = max(BOUNDARY_MIN_COLUMNS_ABS, int(w * BOUNDARY_MIN_COLUMNS_RATIO))
    if len(xs) < min_columns:
        return None

    curve = np.full(w, np.nan, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.float32)
    curve[xs] = ys

    valid = np.where(np.isfinite(curve))[0]
    if valid.size < 2:
        return None

    curve = np.interp(
        np.arange(w, dtype=np.float32),
        valid.astype(np.float32),
        curve[valid],
    ).astype(np.float32)
    curve = median_filter_1d(curve, BOUNDARY_MEDIAN_KERNEL)
    curve = smooth_1d(curve, max(7, int(round(w * BOUNDARY_SMOOTH_RATIO))))
    curve = cv2.GaussianBlur(
        curve.reshape(1, -1),
        (ensure_odd(BOUNDARY_GAUSSIAN_KERNEL), 1),
        0,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    return np.clip(curve, 0, h - 1)


def classify_mode(pred_mask, image_rgb, roi):
    h, w = pred_mask.shape
    top_region = pred_mask[0:int(h * 0.3), :]
    resin_pixel_count = np.sum(top_region == 0)
    total_top_pixels = top_region.size

    if (resin_pixel_count / total_top_pixels) > 0.03:
        return "four_layer"
    else:
        return "three_layer"


def enforce_boundary_order(boundary_curves, height):
    rc = boundary_curves.get("resin_ceramic")
    cb = boundary_curves.get("ceramic_bonding")
    bs = boundary_curves.get("bonding_substrate")

    if rc is not None and cb is not None:
        cb = np.maximum(cb, rc + MIN_BOUNDARY_GAP)
        boundary_curves["ceramic_bonding"] = np.clip(cb, 0, height - 1)

    cb = boundary_curves.get("ceramic_bonding")
    if cb is not None and bs is not None:
        bs = np.maximum(bs, cb + MIN_BOUNDARY_GAP)
        boundary_curves["bonding_substrate"] = np.clip(bs, 0, height - 1)

    return boundary_curves


def build_boundary_curves(pred_mask, image_rgb, roi=None):
    h, _ = pred_mask.shape
    if roi is None:
        roi = (0, 0, pred_mask.shape[1], pred_mask.shape[0])

    mode = classify_mode(pred_mask, image_rgb, roi)
    class_map = get_mode_class_map(mode)

    analysis_mask = mask_out_scale_region(pred_mask, image_rgb)

    if mode == "four_layer":
        boundary_curves = {
            "resin_ceramic": curve_from_layer_top(analysis_mask, class_map["ceramic"], roi=roi),
            "ceramic_bonding": curve_from_layer_top(analysis_mask, class_map["bonding"], roi=roi),
            "bonding_substrate": curve_from_layer_top(analysis_mask, class_map["substrate"], roi=roi),
        }
    else:
        boundary_curves = {
            "resin_ceramic": None,
            "ceramic_bonding": curve_from_layer_top(analysis_mask, class_map["bonding"], roi=roi),
            "bonding_substrate": curve_from_layer_top(analysis_mask, class_map["substrate"], roi=roi),
        }

    return enforce_boundary_order(boundary_curves, h), mode, class_map


def curve_to_points(curve):
    if curve is None:
        return None
    xs = np.arange(curve.size, dtype=np.int32)
    ys = np.clip(np.round(curve), 0, 10**9).astype(np.int32)
    return np.stack([xs, ys], axis=1)


def draw_one_boundary(result, curve, boundary_key, thickness):
    points = curve_to_points(curve)
    if points is None:
        return
    color = BOUNDARY_COLORS[boundary_key]
    cv2.polylines(
        result,
        [points.reshape(-1, 1, 2)],
        isClosed=False,
        color=color,
        thickness=int(thickness),
        lineType=cv2.LINE_AA,
    )

    label_y = int(np.clip(int(points[0, 1]) - 8, 18, result.shape[0] - 8))
    cv2.putText(
        result,
        BOUNDARY_NAMES[boundary_key],
        (8, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_boundary_lines_on_image(base_rgb, boundary_curves, line_thickness=4):
    result = base_rgb.copy()
    found = False
    for boundary_key in ("resin_ceramic", "ceramic_bonding", "bonding_substrate"):
        curve = boundary_curves.get(boundary_key)
        if curve is not None:
            draw_one_boundary(result, curve, boundary_key, int(line_thickness))
            found = True
    return result, found


def build_layer_band_mask(pred_mask, boundary_curves, layer_key, class_map):
    h, w = pred_mask.shape

    if layer_key == "ceramic":
        class_id = class_map.get("ceramic")
        upper = boundary_curves.get("resin_ceramic")
        lower = boundary_curves.get("ceramic_bonding")
        if upper is None and class_id is not None:
            upper = np.zeros(w, dtype=np.float32)
    elif layer_key == "bonding":
        class_id = class_map.get("bonding")
        upper = boundary_curves.get("ceramic_bonding")
        lower = boundary_curves.get("bonding_substrate")
    else:
        return np.zeros((h, w), dtype=np.uint8)

    if class_id is None:
        return np.zeros((h, w), dtype=np.uint8)

    base_mask = clean_layer_mask(pred_mask == class_id)

    if upper is None or lower is None:
        return largest_component_mask(base_mask) if cv2.countNonZero(base_mask) > 0 else base_mask

    band_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        y_top = int(np.clip(round(upper[x]), 0, h - 1))
        y_bottom = int(np.clip(round(lower[x]), 0, h - 1))
        if y_bottom > y_top:
            band_mask[y_top:y_bottom, x] = 255

    combined = cv2.bitwise_and(base_mask, band_mask)
    if cv2.countNonZero(combined) > 0:
        return largest_component_mask(combined)
    if cv2.countNonZero(base_mask) > 0:
        return largest_component_mask(base_mask)
    return largest_component_mask(band_mask)


def get_porosity_target_layer(mode):
    return "ceramic"


def detect_particles_and_pores(image_rgb, target_mask):
    h, w = image_rgb.shape[:2]
    full_pore_mask = np.zeros((h, w), dtype=np.uint8)

    if target_mask is None or cv2.countNonZero(target_mask) == 0:
        return {"pore_ratio": 0.0, "otsu_threshold": 0, "pore_mask": full_pore_mask}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    target_indices = target_mask > 0

    mean_val = float(np.mean(gray[target_indices]))
    masked_gray = gray.copy()
    masked_gray[~target_indices] = int(mean_val)

    kernel_size = int(w / 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(51, kernel_size)

    bg_illumination = cv2.GaussianBlur(masked_gray, (kernel_size, kernel_size), 0).astype(np.float32)
    corrected = (gray.astype(np.float32) / (bg_illumination + 1e-5)) * mean_val
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    target_pixels = corrected[target_indices]
    if len(target_pixels) == 0:
        return {"pore_ratio": 0.0, "otsu_threshold": 0, "pore_mask": full_pore_mask}

    thresh_val, _ = cv2.threshold(target_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pore_in_target = (corrected < thresh_val) & target_indices
    full_pore_mask[pore_in_target] = 255

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    full_pore_mask = cv2.morphologyEx(full_pore_mask, cv2.MORPH_OPEN, clean_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(full_pore_mask, 8)
    filtered = np.zeros_like(full_pore_mask)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= PORE_MIN_COMPONENT_AREA:
            filtered[labels == label_id] = 255
    full_pore_mask = filtered

    total_target_cnt = int(np.sum(target_indices))
    pore_cnt = int(np.sum(full_pore_mask > 0))
    pore_ratio = float(pore_cnt / total_target_cnt * 100.0) if total_target_cnt > 0 else 0.0

    return {"pore_ratio": pore_ratio, "otsu_threshold": int(thresh_val), "pore_mask": full_pore_mask}


def overlay_pore_mask(result_rgb, pore_mask, color=(255, 215, 0), alpha=0.45):
    if pore_mask is None or cv2.countNonZero(pore_mask) == 0:
        return result_rgb
    output = result_rgb.copy()
    indices = pore_mask > 0
    output[indices] = ((1 - alpha) * result_rgb[indices] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return output


def run_tesseract_ocr(image, psm_values=(7, 8, 6, 11)):
    """
    使用多个 PSM 分别执行 OCR。

    注意：
    这里不能在第一个识别到数字后 break。
    例如 psm=7 可能返回 10，但 psm=8 或 psm=6 可能返回 100。
    所有结果都需要保留下来，交给上层做投票和冲突判断。
    """
    if not os.path.isfile(TESSERACT_CMD):
        raise FileNotFoundError(
            f"找不到 OCR 程序：{TESSERACT_CMD}"
        )

    if image is None or image.size == 0:
        return []

    enlarged = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    if enlarged.ndim == 3:
        # 当前项目的图像统一使用 RGB
        enlarged = cv2.cvtColor(
            enlarged,
            cv2.COLOR_RGB2GRAY,
        )

    enlarged = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0,
    )

    _, binary = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # 保证文字通常是黑字白底
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    binary = cv2.copyMakeBorder(
        binary,
        18,
        18,
        24,
        24,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        ok, encoded = cv2.imencode(
            ".png",
            binary,
        )

        if not ok:
            raise RuntimeError(
                "比例尺 OCR 临时图像编码失败。"
            )

        encoded.tofile(temp_path)

        attempts = []

        for psm in psm_values:
            completed = subprocess.run(
                [
                    TESSERACT_CMD,
                    temp_path,
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    str(psm),
                    "-c",
                    (
                        "tessedit_char_whitelist="
                        "0123456789.,uUmM"
                    ),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
                timeout=20,
                check=False,
            )

            text = completed.stdout.strip()

            attempts.append(
                {
                    "psm": int(psm),
                    "text": text,
                    "stderr": completed.stderr.strip(),
                    "returncode": int(completed.returncode),
                }
            )

        return attempts

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

def save_ocr_debug_image(path, image):
    """
    保存 OCR 调试图像。
    """
    if image is None or image.size == 0:
        return

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    output = image

    if output.dtype != np.uint8:
        output = cv2.normalize(
            output,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    if output.ndim == 3:
        output = cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR,
        )

    cv2.imwrite(path, output)


def save_ocr_debug_json(path, data):
    """
    保存 OCR 原始文本、PSM 和解析结果。
    """
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

def parse_scale_value_um(ocr_text):
    normalized = ocr_text.replace("，", ",")
    numbers = re.findall(r"\d+(?:[\.,]\d+)?", normalized)
    if not numbers:
        raise ValueError(f"OCR 未识别出比例尺数值（识别文本：{ocr_text!r}）。")
    values = []
    for number in numbers:
        try:
            value = float(number.replace(",", "."))
        except ValueError:
            continue
        if np.isfinite(value) and value > 0:
            values.append(value)
    if not values:
        raise ValueError(f"比例尺数值无效（识别文本：{ocr_text!r}）。")
    return max(values)


def _collect_horizontal_scale_candidates(binary, color_priority=0):
    crop_h, crop_w = binary.shape
    kernel_width = max(7, int(round(crop_w * 0.025)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    candidates = []
    for component_id in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[component_id]]
        if width < max(14, int(crop_w * 0.025)):
            continue
        if width > int(crop_w * 0.82):
            continue
        if height > max(20, int(crop_h * 0.12)):
            continue
        if width / max(height, 1) < 4.0:
            continue
        if x <= 2 or x + width >= crop_w - 2:
            continue
        if y <= 2 or y + height >= crop_h - 2:
            continue
        score = float(color_priority) + float(width) + 0.12 * float(y) + 0.04 * float(x) + 0.01 * float(area)
        candidates.append({"score": score, "x": x, "y": y, "width": width, "height": height, "pixels": float(width)})
    return candidates


def detect_scale_bar(scale_crop):
    gray = cv2.cvtColor(scale_crop, cv2.COLOR_RGB2GRAY)
    red = scale_crop[:, :, 0].astype(np.int16)
    green = scale_crop[:, :, 1].astype(np.int16)
    blue = scale_crop[:, :, 2].astype(np.int16)

    red_mask = (
        (red >= 95)
        & (red - green >= 28)
        & (red - blue >= 28)
    ).astype(np.uint8) * 255
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )

    _, dark = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, light = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidates = []
    candidates.extend(_collect_horizontal_scale_candidates(red_mask, 10000))
    candidates.extend(_collect_horizontal_scale_candidates(dark, 0))
    candidates.extend(_collect_horizontal_scale_candidates(light, 0))
    if not candidates:
        raise ValueError("没有在图片右下区域检测到水平比例尺线段。")
    
    best = max(candidates, key=lambda item: item["score"])
    # 强制做一次全字段 int 转换，彻底杜绝 float 传递
    return {
        "score": float(best["score"]),
        "x": int(best["x"]),
        "y": int(best["y"]),
        "width": int(best["width"]),
        "height": int(best["height"]),
        "pixels": float(best["pixels"]),
    }


def make_scale_text_regions(scale_crop, bar):
    """
    根据比例尺线生成多个文字 OCR 区域。
    所有切片坐标强制转换为 int，避免 float 切片报错。
    """
    crop_h, crop_w = scale_crop.shape[:2]

    x = int(bar["x"])
    y = int(bar["y"])
    width = int(bar["width"])
    height = int(bar["height"])

    regions = []

    for horizontal_pad_ratio, vertical_ratio in (
        (0.30, 0.42),
        (0.65, 0.65),
        (1.00, 0.90),
        (1.35, 1.15),
    ):
        horizontal_pad = max(20, int(round(width * horizontal_pad_ratio)))
        text_height = max(30, int(round(width * vertical_ratio)))

        x0 = int(max(0, x - horizontal_pad))
        x1 = int(min(crop_w, x + width + horizontal_pad))
        y0 = int(max(0, y - text_height))
        y1 = int(min(crop_h, y + max(4, height // 2)))

        if x1 <= x0 or y1 <= y0:
            continue

        region = scale_crop[y0:y1, x0:x1]

        if region.size > 0:
            regions.append(
                {
                    "image": region,
                    "bbox": [x0, y0, x1, y1],
                }
            )

    return regions


def make_ocr_variants(region):
    variants = [region]
    red = region[:, :, 0].astype(np.int16)
    green = region[:, :, 1].astype(np.int16)
    blue = region[:, :, 2].astype(np.int16)
    red_advantage = np.clip(red - np.maximum(green, blue), 0, 255).astype(np.uint8)
    if int(np.max(red_advantage)) >= 20:
        _, red_binary = cv2.threshold(red_advantage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.insert(0, cv2.bitwise_not(red_binary))
    return variants


def recognize_scale(image_rgb, debug_name=None):
    """
    检测比例尺并使用多个 OCR 结果进行投票。
    """
    h, w = image_rgb.shape[:2]

    x0 = int(round(w * SCALE_ROI_LEFT_RATIO))
    y0 = int(round(h * SCALE_ROI_TOP_RATIO))

    scale_crop = image_rgb[y0:h, x0:w]

    if scale_crop.size == 0:
        raise ValueError("图片尺寸过小，无法截取右下角比例尺区域。")

    bar = detect_scale_bar(scale_crop)

    if debug_name:
        debug_stem = os.path.splitext(os.path.basename(debug_name))[0]
    else:
        debug_stem = "unknown_image"

    debug_dir = os.path.join(OCR_DEBUG_ROOT, debug_stem)

    if OCR_DEBUG_ENABLED:
        os.makedirs(debug_dir, exist_ok=True)
        save_ocr_debug_image(os.path.join(debug_dir, "01_scale_crop.png"), scale_crop)

        bar_debug = scale_crop.copy()
        cv2.rectangle(
            bar_debug,
            (int(bar["x"]), int(bar["y"])),
            (int(bar["x"] + bar["width"]), int(bar["y"] + bar["height"])),
            (0, 255, 0),
            2,
        )
        save_ocr_debug_image(os.path.join(debug_dir, "02_detected_scale_bar.png"), bar_debug)

    attempts = []
    candidate_values = []

    regions = make_scale_text_regions(scale_crop, bar)

    for region_index, region_info in enumerate(regions):
        region = region_info["image"]

        if OCR_DEBUG_ENABLED:
            save_ocr_debug_image(
                os.path.join(debug_dir, f"region_{region_index:02d}_original.png"),
                region,
            )

        variants = make_ocr_variants(region)

        for variant_index, variant in enumerate(variants):
            if OCR_DEBUG_ENABLED:
                save_ocr_debug_image(
                    os.path.join(debug_dir, f"region_{region_index:02d}_variant_{variant_index:02d}.png"),
                    variant,
                )

            ocr_results = run_tesseract_ocr(variant, psm_values=(7, 8, 6, 11))

            for ocr_result in ocr_results:
                text = ocr_result["text"]
                parsed_value = None
                parse_error = None

                if re.search(r"\d", text):
                    try:
                        parsed_value = parse_scale_value_um(text)
                        candidate_values.append(float(parsed_value))
                    except ValueError as exc:
                        parse_error = str(exc)

                attempts.append(
                    {
                        "region_index": int(region_index),
                        "region_bbox": [int(v) for v in region_info["bbox"]],
                        "variant_index": int(variant_index),
                        "psm": int(ocr_result["psm"]),
                        "text": str(text),
                        "parsed_value_um": float(parsed_value) if parsed_value is not None else None,
                        "parse_error": parse_error,
                    }
                )

    # 如果针对 region 未识别出数字，尝试在整个 scale_crop 兜底
    if not candidate_values:
        fallback_variants = make_ocr_variants(scale_crop)
        for variant_index, variant in enumerate(fallback_variants):
            if OCR_DEBUG_ENABLED:
                save_ocr_debug_image(
                    os.path.join(debug_dir, f"fallback_variant_{variant_index:02d}.png"),
                    variant,
                )

            ocr_results = run_tesseract_ocr(variant, psm_values=(11, 6, 7, 8))
            for ocr_result in ocr_results:
                text = ocr_result["text"]
                parsed_value = None
                parse_error = None
                if re.search(r"\d", text):
                    try:
                        parsed_value = parse_scale_value_um(text)
                        candidate_values.append(float(parsed_value))
                    except ValueError as exc:
                        parse_error = str(exc)

                attempts.append(
                    {
                        "region_index": -1,
                        "region_bbox": [0, 0, int(scale_crop.shape[1]), int(scale_crop.shape[0])],
                        "variant_index": int(variant_index),
                        "psm": int(ocr_result["psm"]),
                        "text": str(text),
                        "parsed_value_um": float(parsed_value) if parsed_value is not None else None,
                        "parse_error": parse_error,
                    }
                )

    if not candidate_values:
        recognized = " | ".join(item["text"] for item in attempts if item["text"]) or "<空>"
        if OCR_DEBUG_ENABLED:
            save_ocr_debug_json(
                os.path.join(debug_dir, "ocr_result.json"),
                {
                    "bar": bar,
                    "attempts": attempts,
                    "candidate_values": [],
                    "selected_value_um": None,
                },
            )
        raise ValueError(
            f"OCR 未识别出比例尺数字（多方案结果：{recognized!r}）。Debug目录：{debug_dir}"
        )

    # 统计候选值投票数
    value_counts = {}
    for val in candidate_values:
        val_key = float(val)
        value_counts[val_key] = value_counts.get(val_key, 0) + 1

    # 按照数值从大到小排序候选值（如 100.0, 10.0）
    sorted_values = sorted(value_counts.keys(), reverse=True)
    selected_value_um = sorted_values[0] # 默认选最大有效候选值

    # 如果存在较大值（如 100）与较小值（如 10）竞价
    # 只要较大值的得票率超过 25%，说明它是被完整识别的真实数值，较小值只是笔画残缺导致的误读
    total_votes = len(candidate_values)
    for val in sorted_values:
        if value_counts[val] / total_votes >= 0.25:
            selected_value_um = val
            break

    selected_texts = [
        item["text"]
        for item in attempts
        if item["parsed_value_um"] is not None and float(item["parsed_value_um"]) == selected_value_um
    ]
    selected_text = selected_texts[0] if selected_texts else str(selected_value_um)

    result = {
        "bar": bar,
        "candidate_values": candidate_values,
        "value_counts": value_counts,
        "selected_value_um": selected_value_um,
        "selected_text": selected_text,
        "scale_pixels": float(bar["pixels"]),
        "microns_per_pixel": float(selected_value_um / bar["pixels"]),
    }

    if OCR_DEBUG_ENABLED:
        save_ocr_debug_json(os.path.join(debug_dir, "ocr_result.json"), result)

    print(
        f"【OCR检测成功】数值: {selected_value_um:g} um | 像素长度: {bar['pixels']:.1f} px | 投票表: {value_counts}"
    )

    return (
        selected_value_um / float(bar["pixels"]),
        selected_value_um,
        float(bar["pixels"]),
        selected_text,
    )


def measure_layer_thickness(boundary_curves, roi, microns_per_pixel, mode, sample_count=50):
    rc = boundary_curves.get("resin_ceramic")
    cb = boundary_curves.get("ceramic_bonding")
    bs = boundary_curves.get("bonding_substrate")

    width = len(cb) if cb is not None else len(bs) if bs is not None else 0
    if width == 0:
        return {}

    x0, x1 = (0, width) if roi is None else (roi[0], roi[2])
    if x1 <= x0:
        return {}

    xs_sampled = np.unique(np.rint(np.linspace(x0, x1 - 1, sample_count)).astype(np.int32))
    xs_full = np.arange(x0, x1, dtype=np.int32)

    if mode == "three_layer":
        top_zeros = np.zeros(width, dtype=np.float32)
        layer_specs = (
            ("ceramic", "陶瓷层", top_zeros, cb),
            ("bonding", "粘结层", cb, bs)
        )
    else:
        layer_specs = (
            ("ceramic", "陶瓷层", rc, cb),
            ("bonding", "粘结层", cb, bs)
        )

    measurements = {}
    for key, display_name, upper_curve, lower_curve in layer_specs:
        if upper_curve is None or lower_curve is None:
            continue

        upper_y_s, lower_y_s = upper_curve[xs_sampled], lower_curve[xs_sampled]
        gaps_px_s = lower_y_s - upper_y_s
        valid_s = np.isfinite(gaps_px_s) & (gaps_px_s > 0)
        sampled_res = (
            {
                "xs": xs_sampled[valid_s],
                "upper_y": upper_y_s[valid_s],
                "lower_y": lower_y_s[valid_s],
                "values_um": gaps_px_s[valid_s] * float(microns_per_pixel),
                "min_um": float(np.min(gaps_px_s[valid_s] * float(microns_per_pixel))),
                "max_um": float(np.max(gaps_px_s[valid_s] * float(microns_per_pixel))),
                "mean_um": float(np.mean(gaps_px_s[valid_s] * float(microns_per_pixel))),
                "count": int(np.sum(valid_s)),
            }
            if np.any(valid_s)
            else None
        )

        upper_y_f, lower_y_f = upper_curve[xs_full], lower_curve[xs_full]
        gaps_px_f = lower_y_f - upper_y_f
        valid_f = np.isfinite(gaps_px_f) & (gaps_px_f > 0)
        full_res = (
            {
                "xs": xs_full[valid_f],
                "upper_y": upper_y_f[valid_f],
                "lower_y": lower_y_f[valid_f],
                "values_um": gaps_px_f[valid_f] * float(microns_per_pixel),
                "min_um": float(np.min(gaps_px_f[valid_f] * float(microns_per_pixel))),
                "max_um": float(np.max(gaps_px_f[valid_f] * float(microns_per_pixel))),
                "mean_um": float(np.mean(gaps_px_f[valid_f] * float(microns_per_pixel))),
                "count": int(np.sum(valid_f)),
            }
            if np.any(valid_f)
            else None
        )

        if sampled_res or full_res:
            measurements[key] = {
                "name": display_name,
                "sampled": sampled_res,
                "full": full_res,
                "xs": sampled_res["xs"] if sampled_res else [],
                "upper_y": sampled_res["upper_y"] if sampled_res else [],
                "lower_y": sampled_res["lower_y"] if sampled_res else [],
                "min_um": sampled_res["min_um"] if sampled_res else full_res["min_um"],
                "max_um": sampled_res["max_um"] if sampled_res else full_res["max_um"],
                "mean_um": sampled_res["mean_um"] if sampled_res else full_res["mean_um"],
                "count": sampled_res["count"] if sampled_res else full_res["count"],
            }
    return measurements


def draw_measurement_samples(result, measurements):
    colors = {"ceramic": (255, 215, 0), "bonding": (0, 255, 255)}
    for key, data in measurements.items():
        color = colors.get(key, (255, 255, 255))
        sampled_data = data.get("sampled") or data
        if not sampled_data or "xs" not in sampled_data:
            continue
        for x, upper_y, lower_y in zip(sampled_data["xs"], sampled_data["upper_y"], sampled_data["lower_y"]):
            cv2.line(
                result,
                (int(x), int(round(upper_y))),
                (int(x), int(round(lower_y))),
                color,
                1,
                cv2.LINE_AA,
            )


def resize_for_display(image_rgb, max_w=620, max_h=520):
    h, w = image_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ResUNet34 分层划线与孔隙率识别工具")
        self.root.geometry("1350x850")

        self.model = None
        self.image_path = None
        self.image_rgb = None
        self.pred_mask = None
        self.result_rgb = None
        self.boundary_curves = None
        self.pore_mask = None
        self.roi = None
        self.mode = None
        self.measurements = {}
        self.pore_ratio = None
        self.pore_target_layer = None

        self.show_mask_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value=f"设备：{DEVICE} | 请先导入图片")
        self.measurement_var = tk.StringVar(value="结果：等待处理")
        self.build_ui()

    def build_ui(self):
        top = tk.Frame(self.root, padx=12, pady=10)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="导入图片", width=12, command=self.load_image).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(top, text="开始处理", width=12, command=self.run_predict).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(top, text="保存结果", width=12, command=self.save_result).pack(side=tk.LEFT, padx=(0, 12))

        tk.Checkbutton(
            top,
            text="叠加 Mask 掩膜对比",
            variable=self.show_mask_var,
            command=self.update_display_result,
            font=("Microsoft YaHei", 10, "bold"),
            fg="#1F4E79",
        ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(top, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        measurement_frame = tk.LabelFrame(
            self.root,
            text="统计结果（单位：um / 微米）",
            padx=12,
            pady=7,
            font=("Microsoft YaHei", 11, "bold"),
        )
        measurement_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(
            measurement_frame,
            textvariable=self.measurement_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Microsoft YaHei", 10),
        ).pack(fill=tk.X)

        body = tk.Frame(self.root, padx=12, pady=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left_panel = tk.Frame(body)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right_panel = tk.Frame(body)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        tk.Label(left_panel, text="原图", font=("Microsoft YaHei", 13, "bold")).pack()
        tk.Label(
            right_panel,
            text="预测结果（分层/划线/孔隙率）",
            font=("Microsoft YaHei", 13, "bold"),
        ).pack()

        self.original_label = tk.Label(left_panel, bg="#222222")
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.result_label = tk.Label(right_panel, bg="#222222")
        self.result_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def load_image(self):
        path = filedialog.askopenfilename(
            title="选择待预测图片",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.image_path = path
            self.image_rgb = imread_rgb(path)
            self.pred_mask = None
            self.result_rgb = None
            self.boundary_curves = None
            self.pore_mask = None
            self.measurements = {}
            self.pore_ratio = None
            self.pore_target_layer = None
            self.mode = None
            self.show_original()
            self.result_label.configure(image="", text="")
            self.measurement_var.set("结果：等待处理")
            self.status_var.set(f"已导入：{os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def ensure_model(self):
        if self.model is None:
            self.status_var.set("正在加载 ResUNet34 模型权重，请稍候……")
            self.root.update_idletasks()
            self.model = load_model()

    def run_predict(self):
        if self.image_rgb is None:
            messagebox.showwarning("提示", "请先导入图片。")
            return
        try:
            self.ensure_model()
            self.status_var.set("正在进行层分割、边界提取、孔隙率与厚度计算……")
            self.root.update_idletasks()

            self.pred_mask = predict_outputs(self.model, self.image_rgb)
            self.roi = detect_content_roi(self.image_rgb)
            self.boundary_curves, self.mode, class_map = build_boundary_curves(
                self.pred_mask, self.image_rgb, roi=self.roi
            )

            target_layer = get_porosity_target_layer(self.mode)
            self.pore_mask = None
            if target_layer is not None:
                target_mask = build_layer_band_mask(self.pred_mask, self.boundary_curves, target_layer, class_map)
                if target_mask is not None and cv2.countNonZero(target_mask) > 0:
                    pore_res = detect_particles_and_pores(self.image_rgb, target_mask)
                    self.pore_ratio = pore_res["pore_ratio"]
                    self.pore_target_layer = target_layer
                    self.pore_mask = pore_res.get("pore_mask")

            scale_error = None
            try:
                um_per_pixel, scale_um, scale_pixels, _ = recognize_scale(
                    self.image_rgb,
                    debug_name=self.image_path,
                )
                self.measurements = measure_layer_thickness(
                    self.boundary_curves,
                    self.roi,
                    um_per_pixel,
                    self.mode,
                    THICKNESS_SAMPLE_COUNT,
                )
                self.show_measurements(self.mode, scale_um, scale_pixels, um_per_pixel)
            except Exception as exc:
                scale_error = str(exc)
                self.show_measurements(self.mode, None, None, None, scale_error=scale_error)

            self.update_display_result()

            if scale_error:
                self.status_var.set("边界与孔隙率处理完成，但比例尺识别失败。")
            else:
                self.status_var.set("处理完成：已完成分层划线、Mask对比叠加与统计。")
        except Exception as exc:
            messagebox.showerror("预测失败", str(exc))
            self.status_var.set("预测失败，请检查模型权重及输入图像。")

    def update_display_result(self):
        if self.image_rgb is None or self.boundary_curves is None:
            return

        if self.show_mask_var.get() and self.pred_mask is not None:
            base_canvas = overlay_segmentation_mask(self.image_rgb, self.pred_mask, alpha=0.35)
        else:
            base_canvas = self.image_rgb.copy()

        self.result_rgb, _ = draw_boundary_lines_on_image(base_canvas, self.boundary_curves)

        if self.pore_mask is not None:
            self.result_rgb = overlay_pore_mask(self.result_rgb, self.pore_mask, color=(255, 215, 0), alpha=0.45)

        if self.measurements:
            draw_measurement_samples(self.result_rgb, self.measurements)

        self.show_result()

    def show_measurements(self, mode, scale_um, scale_pixels, um_per_pixel, scale_error=None):
        lines = []
        if mode == "four_layer":
            lines.append("层型判定：4 层 (0:树脂, 1:陶瓷, 2:粘结, 3:基体)")
        elif mode == "three_layer":
            lines.append("层型判定：3 层 (1:陶瓷, 2:粘结, 3:基体)")
        else:
            lines.append("层型判定：未知")

        if scale_error is None and scale_um is not None:
            lines.append(f"比例尺：{scale_um:g} um = {scale_pixels:.1f} px；换算系数：{um_per_pixel:.6f} um/px")
        else:
            lines.append(f"比例尺换算失败：{scale_error}")

        ceramic = self.measurements.get("ceramic")
        bonding = self.measurements.get("bonding")

        if ceramic is not None:
            s, f = ceramic.get("sampled"), ceramic.get("full")
            lines.append(f"【{ceramic['name']}厚度】")
            if s:
                lines.append(f"  · 抽样{s['count']}点: 平均 {s['mean_um']:.3f} um (最小 {s['min_um']:.3f} | 最大 {s['max_um']:.3f})")
            if f:
                lines.append(f"  · 逐列全样({f['count']}列): 平均 {f['mean_um']:.3f} um (最小 {f['min_um']:.3f} | 最大 {f['max_um']:.3f})")

        if bonding is not None:
            s, f = bonding.get("sampled"), bonding.get("full")
            lines.append(f"【{bonding['name']}厚度】")
            if s:
                lines.append(f"  · 抽样{s['count']}点: 平均 {s['mean_um']:.3f} um (最小 {s['min_um']:.3f} | 最大 {s['max_um']:.3f})")
            if f:
                lines.append(f"  · 逐列全样({f['count']}列): 平均 {f['mean_um']:.3f} um (最小 {f['min_um']:.3f} | 最大 {f['max_um']:.3f})")

        if self.pore_ratio is not None:
            lines.append(f"陶瓷层孔隙率：{self.pore_ratio:.2f}%")

        self.measurement_var.set("\n".join(lines))

    def save_result(self):
        if self.result_rgb is None:
            messagebox.showwarning("提示", "请先执行预测。")
            return
        default_name = "prediction_result.png"
        if self.image_path:
            stem = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{stem}_prediction.png"
        path = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            imwrite_rgb(path, self.result_rgb)
            self.status_var.set(f"已保存：{path}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def show_original(self):
        image = resize_for_display(self.image_rgb)
        self.original_photo = ImageTk.PhotoImage(image)
        self.original_label.configure(image=self.original_photo)

    def show_result(self):
        image = resize_for_display(self.result_rgb)
        self.result_photo = ImageTk.PhotoImage(image)
        self.result_label.configure(image=self.result_photo)
        self.result_label.image = self.result_photo


def main():
    root = tk.Tk()
    PredictApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()