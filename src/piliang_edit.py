import importlib.util
import os
import queue
import shutil
import tempfile
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
PAGE_MARGIN = 55
PAGE_HEADER_HEIGHT = 95
ROW_GAP = 22
IMAGE_COLUMN_WIDTH = 700
REPORT_DPI = 150
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(SCRIPT_DIR, "enhanced_boundary_detector.py")
REQUIRED_DETECTOR_VERSION = "2026-07-20-resunet34-layer-boundary-v3"
BATCH_APP_VERSION = "3.1"


def load_current_detector():
    if not os.path.isfile(DETECTOR_PATH):
        raise FileNotFoundError(f"找不到厚度识别核心脚本：{DETECTOR_PATH}")

    module_name = (
        "_enhanced_boundary_detector_batch_"
        f"{os.stat(DETECTOR_PATH).st_mtime_ns}"
    )
    spec = importlib.util.spec_from_file_location(module_name, DETECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载厚度识别核心脚本：{DETECTOR_PATH}")

    runtime_detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_detector)

    actual_version = getattr(runtime_detector, "ALGORITHM_VERSION", "<未标记版本>")
    if actual_version != REQUIRED_DETECTOR_VERSION:
        raise RuntimeError(
            "批量程序检测到核心脚本版本不一致。\n"
            f"要求版本：{REQUIRED_DETECTOR_VERSION}\n"
            f"实际版本：{actual_version}\n"
            f"加载位置：{DETECTOR_PATH}\n\n"
            "请把最新版 enhanced_boundary_detector.py 与批量程序放在同一目录。"
        )
    return runtime_detector


def load_chinese_font(size, bold=False):
    candidates = (
        [
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", # Linux 兼容
            "/System/Library/Fonts/PingFang.ttc"                    # macOS 兼容
        ]
        if bold
        else [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc"
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_TITLE = load_chinese_font(34, bold=True)
FONT_NAME = load_chinese_font(25, bold=True)
FONT_BODY = load_chinese_font(21)
FONT_SMALL = load_chinese_font(17)


def fit_image_to_box(image, box_width, box_height):
    image = image.convert("RGB")
    copy = image.copy()
    copy.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)
    return copy


def wrap_text(draw, text, font, max_width):
    text = str(text)
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            box = draw.textbbox((0, 0), candidate, font=font)
            if current and box[2] - box[0] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return lines


def draw_wrapped_text(draw, position, text, font, fill, max_width, line_spacing=8):
    x, y = position
    line_height = draw.textbbox((0, 0), "测试Ag", font=font)[3]
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_spacing
    return y


def layer_summary(data):
    if data is None:
        return "未识别"
    
    # 获取两种测厚逻辑的数据
    s_data = data.get("sampled")
    f_data = data.get("full")
    
    lines = []
    if s_data:
        lines.append(
            f"[抽样50点] 平均:{s_data['mean_um']:.3f} um (范围:{s_data['min_um']:.3f}~{s_data['max_um']:.3f})"
        )
    if f_data:
        lines.append(
            f"[逐列全样] 平均:{f_data['mean_um']:.3f} um (范围:{f_data['min_um']:.3f}~{f_data['max_um']:.3f})"
        )
    
    if lines:
        return "\n".join(lines)
    
    # 兼容备用
    return (
        f"最小值：{data['min_um']:.3f} um\n"
        f"最大值：{data['max_um']:.3f} um\n"
        f"平均值：{data['mean_um']:.3f} um"
    )


def get_porosity_label(record):
    target = record.get("pore_target_layer")
    if target == "ceramic":
        return "陶瓷层孔隙率"
    if target == "bonding":
        return "粘结层孔隙率"
    return "目标层孔隙率"


def draw_sample_row(page, record, row_index):
    draw = ImageDraw.Draw(page)
    usable_height = PAGE_HEIGHT - PAGE_HEADER_HEIGHT - PAGE_MARGIN * 2
    row_height = (usable_height - ROW_GAP) // 2
    row_top = PAGE_MARGIN + PAGE_HEADER_HEIGHT + row_index * (row_height + ROW_GAP)
    row_bottom = row_top + row_height

    image_left = PAGE_MARGIN
    image_right = image_left + IMAGE_COLUMN_WIDTH
    info_left = image_right + 34
    info_width = PAGE_WIDTH - PAGE_MARGIN - info_left

    draw.rounded_rectangle(
        (PAGE_MARGIN, row_top, PAGE_WIDTH - PAGE_MARGIN, row_bottom),
        radius=14,
        outline=(185, 190, 198),
        width=2,
        fill=(250, 251, 253),
    )
    draw.line(
        (image_right + 16, row_top + 18, image_right + 16, row_bottom - 18),
        fill=(210, 214, 220),
        width=2,
    )

    image_box_width = IMAGE_COLUMN_WIDTH - 35
    image_box_height = row_height - 40
    try:
        with Image.open(record["result_path"]) as source:
            fitted = fit_image_to_box(source, image_box_width, image_box_height)
        paste_x = image_left + (image_box_width - fitted.width) // 2 + 8
        paste_y = row_top + (row_height - fitted.height) // 2
        page.paste(fitted, (paste_x, paste_y))
    except Exception as exc:
        draw_wrapped_text(
            draw,
            (image_left + 24, row_top + 32),
            f"图片载入失败：{exc}",
            FONT_BODY,
            (170, 30, 30),
            image_box_width - 30,
        )

    y = row_top + 27
    y = draw_wrapped_text(
        draw,
        (info_left, y),
        record["name"],
        FONT_NAME,
        (25, 38, 57),
        info_width,
        line_spacing=6,
    )
    y += 8

    mode = record.get("mode")
    if mode == "four_layer":
        mode_text = "层型：4 层（0,1,2,3）"
    elif mode == "three_layer":
        mode_text = "层型：3 层（1,2,3）"
    else:
        mode_text = "层型：未知"
    draw.text((info_left, y), mode_text, font=FONT_SMALL, fill=(80, 87, 98))
    y += 30

    if not record["success"]:
        draw.text(
            (info_left, y),
            "处理失败",
            font=FONT_NAME,
            fill=(190, 35, 35),
        )
        y += 45
        draw_wrapped_text(
            draw,
            (info_left, y),
            record.get("error", "未知错误"),
            FONT_BODY,
            (125, 35, 35),
            info_width,
            line_spacing=7,
        )
        return

    if record.get("scale_um") is not None and record.get("scale_pixels") is not None:
        scale_text = (
            f"比例尺：{record['scale_um']:g} um = "
            f"{record['scale_pixels']:.1f} px"
        )
    else:
        scale_text = "比例尺：未识别"
    draw.text((info_left, y), scale_text, font=FONT_SMALL, fill=(80, 87, 98))
    y += 35

    draw.text((info_left, y), "陶瓷层", font=FONT_NAME, fill=(190, 55, 35))
    y += 35

    ceramic_summary_str = layer_summary(record["measurements"].get("ceramic"))
    if record.get("mode") == "four_layer" and record.get("pore_ratio") is not None:
        ceramic_summary_str += f"\n陶瓷层孔隙率：{record['pore_ratio']:.2f}%"
    elif record.get("mode") == "three_layer":
        ceramic_summary_str = "当前 3 层规则下不计算"

    y = draw_wrapped_text(
        draw,
        (info_left, y),
        ceramic_summary_str,
        FONT_BODY,
        (32, 42, 55),
        info_width,
        line_spacing=5,
    )
    y += 14

    draw.text((info_left, y), "粘结层", font=FONT_NAME, fill=(35, 105, 190))
    y += 35

    bonding_summary_str = layer_summary(record["measurements"].get("bonding"))
    if record.get("mode") == "three_layer" and record.get("pore_ratio") is not None:
        bonding_summary_str += f"\n粘结层孔隙率：{record['pore_ratio']:.2f}%"

    draw_wrapped_text(
        draw,
        (info_left, y),
        bonding_summary_str,
        FONT_BODY,
        (32, 42, 55),
        info_width,
        line_spacing=5,
    )


def build_report_pages(records):
    pages = []
    total_pages = (len(records) + 1) // 2
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for page_index in range(total_pages):
        page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(page)
        draw.text(
            (PAGE_MARGIN, PAGE_MARGIN - 8),
            "样本厚度识别报告",
            font=FONT_TITLE,
            fill=(20, 35, 55),
        )
        page_text = f"生成时间：{generated_at}    第 {page_index + 1}/{total_pages} 页"
        text_box = draw.textbbox((0, 0), page_text, font=FONT_SMALL)
        draw.text(
            (PAGE_WIDTH - PAGE_MARGIN - (text_box[2] - text_box[0]), PAGE_MARGIN + 7),
            page_text,
            font=FONT_SMALL,
            fill=(100, 106, 116),
        )
        draw.line(
            (PAGE_MARGIN, PAGE_MARGIN + 53, PAGE_WIDTH - PAGE_MARGIN, PAGE_MARGIN + 53),
            fill=(50, 105, 170),
            width=3,
        )

        start = page_index * 2
        for row_index, record in enumerate(records[start : start + 2]):
            draw_sample_row(page, record, row_index)
        pages.append(page)
    return pages

def get_unique_output_path(path):
    """
    确保输出 PDF 不覆盖已有文件。
    如果目标已存在，则自动追加 _001、_002 等后缀。
    """
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    filename = os.path.basename(path)
    stem, ext = os.path.splitext(filename)

    if not ext:
        ext = ".pdf"

    candidate = os.path.join(folder, stem + ext)
    counter = 1

    while os.path.exists(candidate):
        candidate = os.path.join(
            folder,
            f"{stem}_{counter:03d}{ext}",
        )
        counter += 1

    return candidate

def export_pdf(records, output_path):
    if not records:
        raise ValueError("没有可导出的样本记录。")
    pages = build_report_pages(records)
    first_page, remaining_pages = pages[0], pages[1:]
    first_page.save(
        output_path,
        "PDF",
        resolution=REPORT_DPI,
        save_all=True,
        append_images=remaining_pages,
        quality=92,
    )
    for page in pages:
        page.close()


class BatchThicknessApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"样本图片批量厚度识别与 PDF 报告 v{BATCH_APP_VERSION}")
        self.root.geometry("1180x740")
        self.root.minsize(980, 640)

        self.folder_path = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择包含样本图片的文件夹。")
        self.progress_var = tk.DoubleVar(value=0)
        self.records = []
        self.processing = False
        self.event_queue = queue.Queue()
        self.temp_dir = tempfile.mkdtemp(prefix="thickness_report_")

        self.build_ui()
        self.root.after(100, self.poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        top = tk.Frame(self.root, padx=14, pady=13)
        top.pack(fill=tk.X)

        tk.Label(top, text="样本文件夹：", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        tk.Entry(
            top,
            textvariable=self.folder_path,
            state="readonly",
            font=("Microsoft YaHei", 10),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 9), ipady=4)

        self.choose_button = tk.Button(
            top, text="选择文件夹", width=12, command=self.choose_folder
        )
        self.choose_button.pack(side=tk.LEFT, padx=(0, 8))

        self.start_button = tk.Button(
            top, text="开始处理", width=12, command=self.start_processing
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.export_button = tk.Button(
            top,
            text="导出报告",
            width=12,
            command=self.choose_export_path,
            state=tk.DISABLED,
        )
        self.export_button.pack(side=tk.LEFT)

        progress_frame = tk.Frame(self.root, padx=14)
        progress_frame.pack(fill=tk.X)
        ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
        ).pack(fill=tk.X)
        tk.Label(
            progress_frame,
            textvariable=self.status_var,
            anchor="w",
            font=("Microsoft YaHei", 9),
        ).pack(fill=tk.X, pady=(6, 8))

        table_frame = tk.Frame(self.root, padx=14, pady=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("index", "name", "mode", "status", "ceramic", "pore_ratio", "bonding")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.table.heading("index", text="序号")
        self.table.heading("name", text="图片名称")
        self.table.heading("mode", text="层型")
        self.table.heading("status", text="处理状态")
        self.table.heading("ceramic", text="陶瓷层平均厚度 (um)")
        self.table.heading("pore_ratio", text="目标层孔隙率 (%)")
        self.table.heading("bonding", text="粘结层平均厚度 (um)")
        self.table.column("index", width=55, anchor=tk.CENTER, stretch=False)
        self.table.column("name", width=270, anchor=tk.W)
        self.table.column("mode", width=110, anchor=tk.CENTER)
        self.table.column("status", width=140, anchor=tk.CENTER)
        self.table.column("ceramic", width=160, anchor=tk.CENTER)
        self.table.column("pore_ratio", width=150, anchor=tk.CENTER)
        self.table.column("bonding", width=160, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        note = (
            "说明：4 层样本输出红绿蓝线、陶瓷/粘结厚度与陶瓷孔隙率；"
            "3 层样本输出绿蓝线、粘结厚度与粘结孔隙率。"
        )
        tk.Label(
            self.root,
            text=note,
            anchor="w",
            fg="#555555",
            padx=14,
            pady=10,
            font=("Microsoft YaHei", 9),
        ).pack(fill=tk.X)

    def choose_folder(self):
        path = filedialog.askdirectory(title="选择包含待处理样本图片的文件夹")
        if path:
            self.folder_path.set(path)
            image_count = len(self.find_images(path))
            self.status_var.set(f"已选择文件夹，共找到 {image_count} 张样本图片。")

    @staticmethod
    def find_images(folder):
        paths = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(path)
        return sorted(paths, key=lambda value: os.path.basename(value).lower())

    def set_processing_state(self, active):
        self.processing = active
        state = tk.DISABLED if active else tk.NORMAL
        self.choose_button.configure(state=state)
        self.start_button.configure(state=state)
        if active:
            self.export_button.configure(state=tk.DISABLED)
        elif self.records:
            self.export_button.configure(state=tk.NORMAL)

    def start_processing(self):
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("提示", "请先选择有效的样本图片文件夹。")
            return

        image_paths = self.find_images(folder)
        if not image_paths:
            messagebox.showwarning("提示", "所选文件夹中没有支持的图片文件。")
            return

        self.records = []
        self.progress_var.set(0)
        for item in self.table.get_children():
            self.table.delete(item)

        self.set_processing_state(True)
        self.status_var.set("正在加载核心脚本与模型，只需一次……")
        threading.Thread(
            target=self.process_worker,
            args=(image_paths,),
            daemon=True,
        ).start()

    def process_worker(self, image_paths):
        try:
            runtime_detector = load_current_detector()
            self.event_queue.put(
                (
                    "status",
                    f"已加载核心算法 {runtime_detector.ALGORITHM_VERSION}，正在加载模型……",
                )
            )
            model = runtime_detector.load_model()
        except Exception as exc:
            error_text = f"核心脚本或模型加载失败：{exc}\n\n{traceback.format_exc()}"
            self.event_queue.put(("fatal", error_text))
            return

        records = []
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            name = os.path.basename(image_path)
            self.event_queue.put(("status", f"正在处理 {index}/{total}：{name}"))
            record = self.process_one(runtime_detector, model, image_path, index)
            records.append(record)
            self.event_queue.put(("record", index, total, record))

        self.event_queue.put(("finished", records))

    def process_one(self, runtime_detector, model, image_path, index):
        name = os.path.basename(image_path)
        result_path = os.path.join(self.temp_dir, f"{index:05d}_result.jpg")
        record = {
            "name": name,
            "source_path": image_path,
            "result_path": result_path,
            "success": False,
            "error": "",
            "measurements": {},
            "pore_ratio": None,
            "pore_target_layer": None,
            "scale_um": None,
            "scale_pixels": None,
            "um_per_pixel": None,
            "algorithm_version": getattr(runtime_detector, "ALGORITHM_VERSION", None),
            "mode": "unknown",
        }

        image_rgb = None
        result_rgb = None

        try:
            image_rgb = runtime_detector.imread_rgb(image_path)
            pred_mask, _ = runtime_detector.predict_outputs(model, image_rgb)

            result_rgb, found, curves, roi, mode, class_map = runtime_detector.draw_boundary_lines(
                image_rgb,
                pred_mask,
            )
            record["mode"] = mode

            if not found:
                raise ValueError("没有检测到稳定分界线。")

            target_layer = runtime_detector.get_porosity_target_layer(mode)
            if target_layer is not None:
                target_mask = runtime_detector.build_layer_band_mask(
                    pred_mask,
                    curves,
                    target_layer,
                    class_map,
                )
                if target_mask is not None and getattr(target_mask, "size", 0) > 0 and target_mask.sum() > 0:
                    pore_res = runtime_detector.detect_particles_and_pores(
                        image_rgb,
                        target_mask,
                    )
                    record["pore_ratio"] = pore_res.get("pore_ratio")
                    record["pore_target_layer"] = target_layer
                    pore_mask = pore_res.get("pore_mask")
                    if pore_mask is not None:
                        result_rgb = runtime_detector.overlay_pore_mask(
                            result_rgb,
                            pore_mask,
                            color=(255, 215, 0),
                            alpha=0.45,
                        )

            um_per_pixel, scale_um, scale_pixels, _ = runtime_detector.recognize_scale(image_rgb)
            record["scale_um"] = scale_um
            record["scale_pixels"] = scale_pixels
            record["um_per_pixel"] = um_per_pixel

            measurements = runtime_detector.measure_layer_thickness(
                curves,
                roi,
                um_per_pixel,
                mode,
                runtime_detector.THICKNESS_SAMPLE_COUNT,
            )
            if not measurements:
                raise ValueError("缺少可配对的上下边界，无法计算厚度。")

            record["measurements"] = measurements

            runtime_detector.draw_measurement_samples(result_rgb, measurements)
            runtime_detector.imwrite_rgb(result_path, result_rgb)
            record["success"] = True
            return record

        except Exception as exc:
            record["error"] = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc(limit=6)}"
            )
            fallback = result_rgb if result_rgb is not None else image_rgb
            if fallback is not None:
                try:
                    runtime_detector.imwrite_rgb(result_path, fallback)
                except Exception:
                    record["result_path"] = image_path
            else:
                record["result_path"] = image_path
            return record

    def poll_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event[0]

                if kind == "status":
                    self.status_var.set(event[1])

                elif kind == "record":
                    _, index, total, record = event
                    self.add_record_to_table(index, record)
                    self.progress_var.set(index * 100.0 / total)

                elif kind == "finished":
                    self.records = event[1]
                    self.set_processing_state(False)
                    success_count = sum(item["success"] for item in self.records)
                    self.status_var.set(
                        f"处理完成：共 {len(self.records)} 张，成功 {success_count} 张，"
                        f"失败 {len(self.records) - success_count} 张。"
                    )
                    messagebox.showinfo(
                        "处理完成",
                        "文件夹处理完成，现在可以点击“导出报告”选择 PDF 保存位置。",
                    )

                elif kind == "fatal":
                    self.set_processing_state(False)
                    self.status_var.set("初始化失败。")
                    messagebox.showerror("无法开始处理", event[1])

        except queue.Empty:
            pass

        self.root.after(100, self.poll_events)

    def add_record_to_table(self, index, record):
        if record["mode"] == "four_layer":
            mode_text = "4层"
        elif record["mode"] == "three_layer":
            mode_text = "3层"
        else:
            mode_text = "未知"

        if record["success"]:
            ceramic = record["measurements"].get("ceramic")
            bonding = record["measurements"].get("bonding")
            pore_ratio = record.get("pore_ratio")

            ceramic_text = f"{ceramic['mean_um']:.3f}" if ceramic else "-"
            pore_text = f"{pore_ratio:.2f}%" if pore_ratio is not None else "未识别"
            bonding_text = f"{bonding['mean_um']:.3f}" if bonding else "-"
            status = "成功"
        else:
            ceramic_text = "-"
            pore_text = "-"
            bonding_text = "-"
            status = "失败"

        self.table.insert(
            "",
            tk.END,
            values=(
                index,
                record["name"],
                mode_text,
                status,
                ceramic_text,
                pore_text,
                bonding_text,
            ),
        )

    def choose_export_path(self):
        if not self.records:
            messagebox.showwarning("提示", "请先完成文件夹处理。")
            return

        default_name = f"样本厚度识别报告_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        path = filedialog.asksaveasfilename(
            title="导出 PDF 报告",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF 文件", "*.pdf")],
        )
        if not path:
            return

        try:
            output_path = get_unique_output_path(path)

            self.status_var.set("正在生成 PDF 报告……")
            self.root.update_idletasks()

            export_pdf(self.records, output_path)

            self.status_var.set(f"报告已导出：{output_path}")
            messagebox.showinfo("导出完成", f"PDF 报告已保存到：\n{output_path}")
        except Exception as exc:
            self.status_var.set("PDF 报告导出失败。")
            messagebox.showerror("导出失败", str(exc))

    def on_close(self):
        if self.processing:
            if not messagebox.askyesno("确认退出", "批量处理仍在进行，确定要退出吗？"):
                return
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    BatchThicknessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()