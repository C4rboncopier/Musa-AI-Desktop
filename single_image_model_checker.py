from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, filedialog, messagebox
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from banana_mapper.detection import (
    _point_in_polygon,
    _records_from_funnel,
    _sahi_disease_inference,
    _sahi_leaf_inference,
    extract_drone_metadata,
    pixel_to_lat_lon,
)


Image.MAX_IMAGE_PIXELS = None


CLASS_COLORS = {
    "full_leaf": (34, 211, 238, 185),
    "cut_leaf": (251, 146, 60, 210),
    "healthy_leaf": (74, 222, 128, 195),
    "diseased_leaf": (239, 68, 68, 220),
    "black_sigatoka": (168, 85, 247, 230),
    "panama": (239, 68, 68, 230),
}

HEAT_WEIGHTS = {
    "black_sigatoka": 1.0,
    "panama": 1.0,
    "diseased_leaf": 0.72,
    "cut_leaf": 0.18,
    "full_leaf": 0.05,
    "healthy_leaf": 0.025,
}


class SingleImageModelChecker:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Single Image Banana AI Checker")
        self.root.geometry("1450x900")
        self.root.minsize(1160, 720)
        self.root.configure(bg="#0f172a")

        self.leaf_model_path: Path | None = None
        self.disease_model_path: Path | None = None
        self.image_path: Path | None = None
        self.metadata = None

        self.original_image: Image.Image | None = None
        self.composited_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self.leaves = []
        self.diseases = []
        self.records = []
        self.raw_disease_inside_count = 0
        self.raw_disease_outside_count = 0

        self.show_heatmap = BooleanVar(value=True)
        self.show_tile_heatmap = BooleanVar(value=False)
        self.confidence = DoubleVar(value=0.50)
        self.slice_size = IntVar(value=512)
        self.slice_overlap = IntVar(value=96)
        self.nms_iou = DoubleVar(value=0.45)
        self.status_text = StringVar(value="Load models and one drone image to begin.")

        self.class_vars = {
            "full_leaf": BooleanVar(value=True),
            "cut_leaf": BooleanVar(value=True),
            "healthy_leaf": BooleanVar(value=True),
            "diseased_leaf": BooleanVar(value=True),
            "black_sigatoka": BooleanVar(value=True),
            "panama": BooleanVar(value=True),
        }

        self.imscale = 1.0
        self.image_x = 0.0
        self.image_y = 0.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.render_job = None

        self.metric_labels: dict[str, tk.Label] = {}
        self._build_ui()
        self._bind_canvas()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#0f172a")
        top.pack(fill=tk.X, padx=18, pady=14)
        for col in range(4):
            top.grid_columnconfigure(col, weight=1, uniform="top")

        self._build_model_card(top, 0)
        self._build_image_card(top, 1)
        self._build_inference_card(top, 2)
        self._build_status_card(top, 3)

        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        canvas_shell = tk.Frame(body, bg="#111827", highlightbackground="#334155", highlightthickness=1)
        canvas_shell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self.canvas = tk.Canvas(canvas_shell, bg="#020617", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_hint = self.canvas.create_text(
            520,
            340,
            text="Load an image, then run AI inference.",
            fill="#64748b",
            font=("Segoe UI", 13),
        )

        right = tk.Frame(body, bg="#111827", highlightbackground="#334155", highlightthickness=1, width=345)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._build_layers_panel(right)
        self._build_metrics_panel(right)

    def _build_model_card(self, parent: tk.Frame, col: int) -> None:
        card = self._card(parent, "STEP 1: MODELS")
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 8))

        self.leaf_btn = self._button(card, "Load YOLOv8-seg Leaf Model", self.load_leaf_model, "#2563eb")
        self.leaf_btn.pack(fill=tk.X, padx=12, pady=(0, 7))
        self.leaf_label = self._small_label(card, "Leaf model: None")
        self.leaf_label.pack(anchor="w", padx=12, pady=(0, 8))

        self.disease_btn = self._button(card, "Load YOLOv8 Disease Model", self.load_disease_model, "#7c3aed")
        self.disease_btn.pack(fill=tk.X, padx=12, pady=(0, 7))
        self.disease_label = self._small_label(card, "Disease model: None")
        self.disease_label.pack(anchor="w", padx=12, pady=(0, 12))

    def _build_image_card(self, parent: tk.Frame, col: int) -> None:
        card = self._card(parent, "STEP 2: IMAGE")
        card.grid(row=0, column=col, sticky="nsew", padx=8)

        self.image_btn = self._button(card, "Load Single Drone Image", self.load_image, "#475569")
        self.image_btn.pack(fill=tk.X, padx=12, pady=(0, 7))
        self.image_label = self._small_label(card, "Image: None")
        self.image_label.pack(anchor="w", padx=12, pady=(0, 10))

        tk.Checkbutton(
            card,
            text="Show radial heatmap",
            variable=self.show_heatmap,
            command=self.rebuild_overlay,
            bg="#111827",
            fg="#dbeafe",
            selectcolor="#0f172a",
            activebackground="#111827",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=10, pady=(0, 5))

        tk.Checkbutton(
            card,
            text="Show tile heatmap",
            variable=self.show_tile_heatmap,
            command=self.rebuild_overlay,
            bg="#111827",
            fg="#dbeafe",
            selectcolor="#0f172a",
            activebackground="#111827",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def _build_inference_card(self, parent: tk.Frame, col: int) -> None:
        card = self._card(parent, "STEP 3: INFERENCE")
        card.grid(row=0, column=col, sticky="nsew", padx=8)

        row = tk.Frame(card, bg="#111827")
        row.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(row, text="Confidence", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self.confidence, width=7, bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0").pack(side=tk.RIGHT)

        row2 = tk.Frame(card, bg="#111827")
        row2.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(row2, text="Slice size", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Entry(row2, textvariable=self.slice_size, width=7, bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0").pack(side=tk.RIGHT)

        row3 = tk.Frame(card, bg="#111827")
        row3.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(row3, text="Slice overlap", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Entry(row3, textvariable=self.slice_overlap, width=7, bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0").pack(side=tk.RIGHT)

        row4 = tk.Frame(card, bg="#111827")
        row4.pack(fill=tk.X, padx=12, pady=(0, 10))
        tk.Label(row4, text="NMS IoU", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Entry(row4, textvariable=self.nms_iou, width=7, bg="#020617", fg="#e2e8f0", insertbackground="#e2e8f0").pack(side=tk.RIGHT)

        self.run_btn = self._button(card, "Run Single Image Check", self.start_inference, "#10b981")
        self.run_btn.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.save_btn = self._button(card, "Save Result JSON", self.save_results, "#334155")
        self.save_btn.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _build_status_card(self, parent: tk.Frame, col: int) -> None:
        card = self._card(parent, "STATUS")
        card.grid(row=0, column=col, sticky="nsew", padx=(8, 0))
        tk.Label(
            card,
            textvariable=self.status_text,
            bg="#111827",
            fg="#cbd5e1",
            wraplength=280,
            justify=tk.LEFT,
            font=("Segoe UI", 10),
        ).pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def _build_layers_panel(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="CLASS VIEW", bg="#111827", fg="#94a3b8", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(18, 8))

        labels = [
            ("full_leaf", "Full leaf polygons"),
            ("cut_leaf", "Cut leaf polygons"),
            ("healthy_leaf", "Healthy full leaves"),
            ("diseased_leaf", "Diseased full leaves"),
            ("black_sigatoka", "Black sigatoka boxes"),
            ("panama", "Panama boxes"),
        ]
        for key, text in labels:
            cb = tk.Checkbutton(
                parent,
                text=text,
                variable=self.class_vars[key],
                command=self.rebuild_overlay,
                bg="#111827",
                fg="#dbeafe",
                selectcolor="#0f172a",
                activebackground="#111827",
                activeforeground="#ffffff",
                font=("Segoe UI", 10),
            )
            cb.pack(anchor="w", padx=18, pady=3)

        self._separator(parent)

    def _build_metrics_panel(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="METRICS", bg="#111827", fg="#94a3b8", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(12, 8))

        metrics = [
            "Full leaves",
            "Cut leaves",
            "Healthy full leaves",
            "Diseased full leaves",
            "Black sigatoka boxes",
            "Panama boxes",
            "Disease boxes inside leaf",
            "Disease boxes outside leaf",
            "Image size",
            "GPS center",
            "Altitude",
            "Yaw",
            "Ground footprint",
            "GSD",
        ]
        for name in metrics:
            self.metric_labels[name] = self._metric_row(parent, name, "--")

    def _card(self, parent: tk.Frame, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg="#111827", highlightbackground="#334155", highlightthickness=1)
        tk.Label(card, text=title, bg="#111827", fg="#94a3b8", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(12, 8))
        return card

    def _button(self, parent: tk.Frame, text: str, command, bg: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="white",
            activebackground=bg,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            pady=8,
        )

    def _small_label(self, parent: tk.Frame, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg="#111827", fg="#cbd5e1", font=("Segoe UI", 8), wraplength=285, justify=tk.LEFT)

    def _metric_row(self, parent: tk.Frame, key: str, value: str) -> tk.Label:
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill=tk.X, padx=18, pady=4)
        tk.Label(row, text=key + ":", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        label = tk.Label(row, text=value, bg="#111827", fg="#38bdf8", font=("Segoe UI", 9, "bold"), justify=tk.RIGHT)
        label.pack(side=tk.RIGHT)
        return label

    def _separator(self, parent: tk.Frame) -> None:
        tk.Frame(parent, bg="#334155", height=1).pack(fill=tk.X, padx=16, pady=14)

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.root.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Button-4>", self.on_mousewheel)
        self.root.bind("<Button-5>", self.on_mousewheel)

    def load_leaf_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self.leaf_model_path = Path(path)
        self.leaf_label.config(text=f"Leaf model: {self.leaf_model_path.name}", fg="#86efac")

    def load_disease_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self.disease_model_path = Path(path)
        self.disease_label.config(text=f"Disease model: {self.disease_model_path.name}", fg="#86efac")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return

        self.image_path = Path(path)
        self.original_image = Image.open(self.image_path).convert("RGBA")
        self.metadata = extract_drone_metadata(self.image_path)
        self.image_label.config(text=f"Image: {self.image_path.name}", fg="#86efac")

        self.leaves = []
        self.diseases = []
        self.records = []
        self.raw_disease_inside_count = 0
        self.raw_disease_outside_count = 0
        self.composited_image = self.original_image.copy()
        self._reset_detection_metrics()
        self._update_metadata_metrics()
        self._reset_viewport()
        self.render_canvas(False)

        if self.metadata is None:
            self.status_text.set("Image loaded, but required GPS/altitude/focal metadata was not found.")
        else:
            self.status_text.set("Image and embedded drone metadata loaded.")

    def start_inference(self) -> None:
        if not self.leaf_model_path or not self.disease_model_path or not self.image_path or self.original_image is None:
            messagebox.showerror("Missing input", "Please load both models and one image first.")
            return
        if self.metadata is None:
            messagebox.showerror("Missing metadata", "The image is missing GPS, altitude, or focal metadata required for coordinates.")
            return

        try:
            conf = float(self.confidence.get())
            slice_size = int(self.slice_size.get())
            slice_overlap = int(self.slice_overlap.get())
            nms_iou = float(self.nms_iou.get())
        except Exception:
            messagebox.showerror("Invalid settings", "Settings must be numeric.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.status_text.set("Loading models and running SAHI inference...")
        threading.Thread(target=self._inference_worker, args=(conf, slice_size, slice_overlap, nms_iou), daemon=True).start()

    def _class_visible(self, key: str) -> bool:
        var = self.class_vars.get(key)
        return bool(var and var.get())

    def _inference_worker(self, conf: float, slice_size: int, slice_overlap: int, nms_iou: float) -> None:
        try:
            from ultralytics import YOLO

            leaf_model = YOLO(str(self.leaf_model_path))
            disease_model = YOLO(str(self.disease_model_path))

            leaves = _sahi_leaf_inference(
                leaf_model, self.image_path,
                slice_size=slice_size, overlap=slice_overlap,
                conf=conf, iou_threshold=nms_iou,
            )
            diseases = _sahi_disease_inference(
                disease_model, self.image_path,
                slice_size=slice_size, overlap=slice_overlap,
                conf=conf, iou_threshold=nms_iou,
            )

            records = _records_from_funnel(self.image_path.name, self.metadata, leaves, diseases)
            inside, outside = self._count_disease_inside(leaves, diseases)

            self.root.after(0, self._finish_inference, leaves, diseases, records, inside, outside)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _finish_inference(self, leaves, diseases, records, inside: int, outside: int) -> None:
        self.leaves = leaves
        self.diseases = diseases
        self.records = records
        self.raw_disease_inside_count = inside
        self.raw_disease_outside_count = outside
        self.run_btn.config(state=tk.NORMAL)
        self._update_detection_metrics()
        self.rebuild_overlay()
        self.status_text.set("Inference complete. Toggle classes or heatmap to inspect model accuracy.")

    def _show_error(self, message: str) -> None:
        self.run_btn.config(state=tk.NORMAL)
        self.status_text.set("Inference failed.")
        messagebox.showerror("Inference error", message)

    def rebuild_overlay(self) -> None:
        if self.original_image is None:
            return
        self.composited_image = self._build_composited_image()
        self.render_canvas(False)

    def _build_composited_image(self) -> Image.Image:
        base = self.original_image.copy()
        if self.show_heatmap.get() and self.records:
            base = self._apply_heatmap(base)

        if getattr(self, "show_tile_heatmap", None) and self.show_tile_heatmap.get() and self.records:
            base = self._apply_tile_heatmap(base)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for leaf in self.leaves:
            if leaf.class_name == "full_leaf":
                health_key = "diseased_leaf" if leaf.health == "diseased" else "healthy_leaf"
                if not (self.class_vars["full_leaf"].get() or self.class_vars[health_key].get()):
                    continue
                color = CLASS_COLORS[health_key if self.class_vars[health_key].get() else "full_leaf"]
            elif leaf.class_name == "cut_leaf":
                if not self.class_vars["cut_leaf"].get():
                    continue
                color = CLASS_COLORS["cut_leaf"]
            else:
                continue

            polygon = [(float(x), float(y)) for x, y in leaf.polygon]
            if len(polygon) >= 3:
                fill = (color[0], color[1], color[2], 38)
                draw.polygon(polygon, outline=color, fill=fill)
                draw.line(polygon + [polygon[0]], fill=color, width=4)
            x, y = leaf.center
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

        for disease in self.diseases:
            if not self._class_visible(disease.class_name):
                continue
            color = CLASS_COLORS.get(disease.class_name, (255, 255, 255, 220))
            x1, y1, x2, y2 = disease.bbox
            draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
            cx, cy = disease.center
            draw.line((cx - 10, cy, cx + 10, cy), fill=color, width=3)
            draw.line((cx, cy - 10, cx, cy + 10), fill=color, width=3)

        return Image.alpha_composite(base, overlay)

    def _apply_tile_heatmap(self, base: Image.Image) -> Image.Image:
        tile_size = self.slice_size.get()
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        drawn_tiles = set()

        for leaf in self.leaves:
            if leaf.class_name == "full_leaf":
                health_key = "diseased_leaf" if leaf.health == "diseased" else "healthy_leaf"
                if not (self.class_vars["full_leaf"].get() or self.class_vars[health_key].get()):
                    continue
                color = CLASS_COLORS[health_key if self.class_vars[health_key].get() else "full_leaf"]
            elif leaf.class_name == "cut_leaf":
                if not self.class_vars["cut_leaf"].get():
                    continue
                color = CLASS_COLORS["cut_leaf"]
            else:
                continue

            # ~55% Opacity (140 out of 255)
            fill_color = (color[0], color[1], color[2], 140)
            cx, cy = leaf.center
            tx, ty = int(cx // tile_size) * tile_size, int(cy // tile_size) * tile_size
            
            # Prevent over-drawing the same grid block to maintain consistent opacity
            tile_key = (tx, ty, leaf.class_name)
            if tile_key not in drawn_tiles:
                draw.rectangle((tx, ty, tx + tile_size, ty + tile_size), fill=fill_color)
                drawn_tiles.add(tile_key)

        for disease in self.diseases:
            if not self._class_visible(disease.class_name):
                continue
            color = CLASS_COLORS.get(disease.class_name, (255, 255, 255, 255))
            fill_color = (color[0], color[1], color[2], 140)
            cx, cy = disease.center
            tx, ty = int(cx // tile_size) * tile_size, int(cy // tile_size) * tile_size
            tile_key = (tx, ty, disease.class_name)
            if tile_key not in drawn_tiles:
                draw.rectangle((tx, ty, tx + tile_size, ty + tile_size), fill=fill_color)
                drawn_tiles.add(tile_key)

        return Image.alpha_composite(base, overlay)

    def _apply_heatmap(self, base: Image.Image) -> Image.Image:
        w, h = base.size
        scale = min(1.0, 1500.0 / max(w, h))
        hw, hh = max(1, int(w * scale)), max(1, int(h * scale))
        heat = Image.new("L", (hw, hh), 0)
        draw = ImageDraw.Draw(heat)

        points = self._visible_heat_points()
        if not points:
            return base

        radius = max(18, int(min(hw, hh) * 0.035))
        for x, y, weight in points:
            sx, sy = x * scale, y * scale
            intensity = int(255 * max(0.03, min(1.0, weight)))
            draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=intensity)

        heat = heat.filter(ImageFilter.GaussianBlur(radius=max(8, radius // 2)))
        heat = heat.resize((w, h), Image.Resampling.BILINEAR)

        alpha = np.asarray(heat, dtype=np.float32) / 255.0
        alpha = np.power(alpha, 0.68)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = self._heat_colors(alpha)
        rgba[..., 3] = np.clip(alpha * 185, 0, 185).astype(np.uint8)

        heat_img = Image.fromarray(rgba, "RGBA")
        return Image.alpha_composite(base, heat_img)

    def _visible_heat_points(self) -> list[tuple[float, float, float]]:
        disease_layers_visible = (
            self._class_visible("black_sigatoka")
            or self._class_visible("panama")
            or self._class_visible("diseased_leaf")
        )
        points = []

        for disease in self.diseases:
            if self._class_visible(disease.class_name):
                points.append((disease.center[0], disease.center[1], HEAT_WEIGHTS.get(disease.class_name, 1.0)))

        for record in self.records:
            keys = record.layer_keys or [record.class_name]
            if record.class_name in {"black_sigatoka", "panama"}:
                continue
            if not any(self._class_visible(key) for key in keys):
                continue

            is_disease = any(key in {"black_sigatoka", "panama", "diseased_leaf"} for key in keys)
            if disease_layers_visible and not is_disease:
                continue

            weight = max(HEAT_WEIGHTS.get(key, 0.05) for key in keys)
            points.append((record.pixel_x, record.pixel_y, weight))
        return points

    def _heat_colors(self, alpha: np.ndarray) -> np.ndarray:
        stops = [
            (0.00, np.array([34, 211, 238], dtype=np.float32)),
            (0.28, np.array([34, 197, 94], dtype=np.float32)),
            (0.55, np.array([250, 204, 21], dtype=np.float32)),
            (0.78, np.array([249, 115, 22], dtype=np.float32)),
            (1.00, np.array([220, 38, 38], dtype=np.float32)),
        ]
        out = np.zeros((*alpha.shape, 3), dtype=np.float32)
        for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
            mask = (alpha >= a0) & (alpha <= a1)
            if np.any(mask) and a1 > a0:
                t = ((alpha[mask] - a0) / (a1 - a0))[:, None]
                out[mask] = c0 + (c1 - c0) * t
        out[alpha > stops[-1][0]] = stops[-1][1]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _count_disease_inside(self, leaves, diseases) -> tuple[int, int]:
        full_leaves = [leaf for leaf in leaves if leaf.class_name == "full_leaf"]
        inside = 0
        outside = 0
        for disease in diseases:
            matched = any(_point_in_polygon(disease.center, leaf.polygon) for leaf in full_leaves)
            if matched:
                inside += 1
            else:
                outside += 1
        return inside, outside

    def _update_metadata_metrics(self) -> None:
        if self.original_image is not None:
            self.metric_labels["Image size"].config(text=f"{self.original_image.width:,} x {self.original_image.height:,}")
        if self.metadata is None:
            for key in ["GPS center", "Altitude", "Yaw", "Ground footprint", "GSD"]:
                self.metric_labels[key].config(text="N/A", fg="#f87171")
            return

        self.metric_labels["GPS center"].config(text=f"{self.metadata.latitude:.7f}, {self.metadata.longitude:.7f}", fg="#38bdf8")
        self.metric_labels["Altitude"].config(text=f"{self.metadata.altitude_m:.2f} m", fg="#38bdf8")
        self.metric_labels["Yaw"].config(text=f"{self.metadata.yaw_degrees:.2f} deg", fg="#38bdf8")
        self.metric_labels["Ground footprint"].config(
            text=f"{self.metadata.ground_width_m:.2f} x {self.metadata.ground_height_m:.2f} m",
            fg="#38bdf8",
        )
        gsd_cm = (self.metadata.ground_width_m / max(1, self.metadata.image_width)) * 100.0
        self.metric_labels["GSD"].config(text=f"{gsd_cm:.2f} cm/px", fg="#38bdf8")

    def _update_detection_metrics(self) -> None:
        full = sum(1 for leaf in self.leaves if leaf.class_name == "full_leaf")
        cut = sum(1 for leaf in self.leaves if leaf.class_name == "cut_leaf")
        healthy = sum(1 for leaf in self.leaves if leaf.class_name == "full_leaf" and leaf.health == "healthy")
        diseased = sum(1 for leaf in self.leaves if leaf.class_name == "full_leaf" and leaf.health == "diseased")
        sigatoka = sum(1 for disease in self.diseases if disease.class_name == "black_sigatoka")
        panama = sum(1 for disease in self.diseases if disease.class_name == "panama")

        values = {
            "Full leaves": full,
            "Cut leaves": cut,
            "Healthy full leaves": healthy,
            "Diseased full leaves": diseased,
            "Black sigatoka boxes": sigatoka,
            "Panama boxes": panama,
            "Disease boxes inside leaf": self.raw_disease_inside_count,
            "Disease boxes outside leaf": self.raw_disease_outside_count,
        }
        for key, value in values.items():
            self.metric_labels[key].config(text=f"{value:,}", fg="#38bdf8")

    def _reset_detection_metrics(self) -> None:
        for key in [
            "Full leaves",
            "Cut leaves",
            "Healthy full leaves",
            "Diseased full leaves",
            "Black sigatoka boxes",
            "Panama boxes",
            "Disease boxes inside leaf",
            "Disease boxes outside leaf",
        ]:
            self.metric_labels[key].config(text="--", fg="#38bdf8")

    def _reset_viewport(self) -> None:
        if self.original_image is None:
            return
        self.canvas.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.imscale = min(cw / self.original_image.width, ch / self.original_image.height) * 0.95
        self.imscale = max(0.01, self.imscale)
        self.image_x = self.original_image.width / 2
        self.image_y = self.original_image.height / 2

    def start_pan(self, event) -> None:
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event) -> None:
        if self.composited_image is None:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.image_x -= dx / self.imscale
        self.image_y -= dy / self.imscale
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.render_canvas(True)

    def on_mousewheel(self, event) -> None:
        if self.composited_image is None:
            return
        direction = 1 if (hasattr(event, "delta") and event.delta > 0) or getattr(event, "num", None) == 4 else -1
        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_mx = self.image_x + (event.x - cw / 2) / self.imscale
        img_my = self.image_y + (event.y - ch / 2) / self.imscale

        self.imscale *= 1.15 if direction > 0 else 1 / 1.15
        self.imscale = max(0.015, min(self.imscale, 24.0))

        self.image_x = img_mx - (event.x - cw / 2) / self.imscale
        self.image_y = img_my - (event.y - ch / 2) / self.imscale
        self.render_canvas(True)

    def render_canvas(self, fast: bool = True) -> None:
        active = self.composited_image or self.original_image
        if active is None:
            return

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        view_w, view_h = cw / self.imscale, ch / self.imscale
        left, top = self.image_x - view_w / 2, self.image_y - view_h / 2
        right, bottom = self.image_x + view_w / 2, self.image_y + view_h / 2

        crop_left = max(0, int(left))
        crop_top = max(0, int(top))
        crop_right = min(active.width, int(right))
        crop_bottom = min(active.height, int(bottom))
        if crop_left >= crop_right or crop_top >= crop_bottom:
            return

        cropped = active.crop((crop_left, crop_top, crop_right, crop_bottom))
        screen_w = max(1, int((crop_right - crop_left) * self.imscale))
        screen_h = max(1, int((crop_bottom - crop_top) * self.imscale))
        resample = Image.Resampling.NEAREST if fast else Image.Resampling.BILINEAR
        self.tk_image = ImageTk.PhotoImage(cropped.resize((screen_w, screen_h), resample).convert("RGB"))

        draw_x = cw / 2 + ((crop_left + crop_right) / 2 - self.image_x) * self.imscale
        draw_y = ch / 2 + ((crop_top + crop_bottom) / 2 - self.image_y) * self.imscale

        self.canvas.delete("all")
        self.canvas.create_image(draw_x, draw_y, anchor=tk.CENTER, image=self.tk_image)

    def save_results(self) -> None:
        if not self.records or not self.image_path:
            messagebox.showinfo("No results", "Run inference before saving results.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{self.image_path.stem}_single_image_results.json",
        )
        if not path:
            return
        payload = {
            "image": str(self.image_path),
            "metadata": asdict(self.metadata) if self.metadata else None,
            "records": [asdict(record) for record in self.records],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        messagebox.showinfo("Saved", f"Results saved to:\n{path}")


def main() -> None:
    root = tk.Tk()
    SingleImageModelChecker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
