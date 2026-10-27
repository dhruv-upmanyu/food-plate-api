"""Meal photo -> every dish on the plate -> name, portion and nutrition per item and for the whole meal.

    from plate.pipeline import PlateAnalyzer
    meal = PlateAnalyzer().analyze("thali.jpg")
    for item in meal.items: print(item.name, item.count, item.grams, item.kcal)
    print(meal.totals)

Steps
  1. detect: the YOLO food-item detector finds a box around each dish / bowl / piece;
  2. identify: each box (with a little context) goes through the frozen backbone + trained classifier;
  3. portion: countable dishes (roti, idli, egg, ...) are counted box by box; other dishes count one serving
     (katori / glass / plate, see plate/taxonomy.py) per box;
  4. nutrition: grams x per-100 g values from data/nutrition/dish_nutrition.csv, summed for the meal.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from plate.boxes import area, intersection, nms
from plate.imaging import crop_box, square
from plate.features import Backbone
from plate.taxonomy import BY_KEY

ROOT = Path(__file__).resolve().parents[1]
DET_PATH = ROOT / "models" / "yolo" / "detector.pt"
CLS_DIR = ROOT / "models" / "plate_classifier"
NUTRITION = ROOT / "data" / "nutrition" / "dish_nutrition.csv"
FIELDS = ["kcal", "protein_g", "carb_g", "fat_g", "fibre_g"]


@dataclass
class Item:
    dish: str
    name: str
    confidence: float
    alternatives: list[tuple[str, float]]
    boxes: list[list[float]]            # normalised xyxy, one per counted piece / container
    count: float                        # pieces for countable dishes, servings otherwise
    unit: str
    grams: float
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    fibre_g: float
    unsure: bool = False


@dataclass
class Meal:
    items: list[Item]
    totals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"items": [asdict(i) for i in self.items], "totals": self.totals}


def check_not_lfs_pointers(*paths: Path) -> None:
    """A clone made without git-lfs holds small text pointers instead of the models; say so plainly."""
    for p in paths:
        with open(p, "rb") as f:
            if f.read(40).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"{p} is a Git LFS pointer, not the model file: install git-lfs, then run "
                                   "'git lfs install' and 'git lfs pull' in the repository")


def restore_pil_open() -> None:
    """Ultralytics replaces PIL's Image.open, process-wide, with one that pip-installs pi-heif (or raises
    ModuleNotFoundError when auto-install is off) on any file PIL can't identify. Without pi-heif that can only
    fail, so put PIL's own back: unreadable files then raise the usual UnidentifiedImageError."""
    import importlib.util

    if importlib.util.find_spec("pi_heif") is None:
        from ultralytics.utils import patches

        Image.open = getattr(patches, "_image_open", Image.open)


def load_image(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return ImageOps.exif_transpose(image).convert("RGB")


class PlateAnalyzer:
    def __init__(self, det_path: Path = DET_PATH, cls_dir: Path = CLS_DIR, det_conf: float = 0.25,
                 unsure_below: float = 0.4):
        from ultralytics import YOLO

        restore_pil_open()
        meta = json.loads((cls_dir / "meta.json").read_text(encoding="utf-8"))
        weights = [cls_dir / meta["backbone_dir"] / "model.safetensors"] if meta.get("backbone_dir") else []
        check_not_lfs_pointers(det_path, cls_dir / "head.pt", *weights)
        self.detector = YOLO(str(det_path))
        self.labels = meta["labels"]
        self.temperature = meta["temperature"]
        self.scale = meta.get("logit_scale", 30)
        local = cls_dir / meta["backbone_dir"] if meta.get("backbone_dir") else None
        self.backbone = Backbone(meta["backbone"], local_dir=local)
        dim = int(self.backbone.embed([Image.new("RGB", (64, 64))]).shape[1])
        self.head = torch.nn.Linear(dim, len(self.labels))
        self.head.load_state_dict(torch.load(cls_dir / "head.pt", map_location="cpu"))
        self.head.eval()
        self.nutrition = pd.read_csv(NUTRITION).set_index("key")
        self.det_conf = det_conf
        self.unsure_below = unsure_below

    # -- steps ---------------------------------------------------------------------------------
    def detect(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        res = self.detector.predict(img, conf=self.det_conf, iou=0.5, imgsz=640, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        boxes = res.boxes.xyxyn.numpy().astype(float)
        conf = res.boxes.conf.numpy().astype(float)
        keep = nms(boxes, conf, 0.6)
        boxes, conf = boxes[keep], conf[keep]
        # a box holding two or more other boxes is a plate / tray, not a dish
        outer = [i for i in range(len(boxes))
                 if sum(1 for j in range(len(boxes)) if j != i and intersection(boxes[i], boxes[j]) >= 0.8 * area(boxes[j])) >= 2]
        keep = [i for i in range(len(boxes)) if i not in outer]
        return boxes[keep], conf[keep]

    def classify(self, img: Image.Image, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, len(self.labels)))
        emb = self.backbone.embed([square(crop_box(img, list(b))) for b in boxes])
        with torch.no_grad():
            return (self.head(torch.tensor(emb)) * self.scale / self.temperature).softmax(-1).numpy()

    def drop_nested_same_dish(self, boxes: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A box that contains another box of the same dish (a plate around the food, a bowl around a smaller
        detection of its contents) would count that dish twice: keep only the inner box."""
        dish = probs.argmax(1) if len(probs) else np.zeros(0, int)
        drop = {i for i in range(len(boxes)) for j in range(len(boxes))
                if i != j and dish[i] == dish[j] and area(boxes[i]) > area(boxes[j])
                and intersection(boxes[i], boxes[j]) >= 0.8 * area(boxes[j])}
        keep = [i for i in range(len(boxes)) if i not in drop]
        return boxes[keep], probs[keep]

    def item(self, dish: str, boxes: list, probs: list[np.ndarray]) -> Item:
        d = BY_KEY[dish]
        n = self.nutrition.loc[dish]
        count = float(len(boxes))
        grams = count * d.unit_g
        mean = np.mean(probs, axis=0)
        order = np.argsort(-mean)[:3]
        conf = float(mean[order[0]])
        values = {f: round(float(n[f]) * grams / 100, 1) for f in FIELDS}
        return Item(dish=dish, name=d.name, confidence=round(conf, 3),
                    alternatives=[(BY_KEY[self.labels[i]].name, round(float(mean[i]), 3)) for i in order[1:]],
                    boxes=[[round(float(v), 4) for v in b] for b in boxes], count=count, unit=d.unit,
                    grams=round(grams), unsure=conf < self.unsure_below, **values)

    def analyze(self, image) -> Meal:
        img = load_image(image)
        boxes, _ = self.detect(img)
        probs = self.classify(img, boxes)
        boxes, probs = self.drop_nested_same_dish(boxes, probs)
        groups: dict[str, list[int]] = {}
        items = []
        for i, p in enumerate(probs):
            dish = self.labels[int(p.argmax())]
            if BY_KEY[dish].countable:          # pieces of one countable dish become one item with a count
                groups.setdefault(dish, []).append(i)
            else:                                # every bowl / glass / plate of a dish is its own serving
                items.append(self.item(dish, [boxes[i]], [p]))
        for dish, idxs in groups.items():
            items.append(self.item(dish, [boxes[i] for i in idxs], [probs[i] for i in idxs]))
        items.sort(key=lambda it: -it.kcal)
        totals = {f: round(sum(getattr(it, f) for it in items), 1) for f in FIELDS}
        totals["grams"] = round(sum(it.grams for it in items))
        return Meal(items=items, totals=totals)
