"""Box utilities: cleaning raw detector boxes and converting grid-cell box specs used in labelling."""
from __future__ import annotations

import re

import numpy as np

GRID = 10  # labelling grid: columns A-J, rows 1-10


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b) -> float:
    inter = intersection(a, b)
    return inter / max(area(a) + area(b) - inter, 1e-9)


def clean_boxes(boxes: list, scores: list, min_score: float = 0.22, max_area: float = 0.85, iou_thr: float = 0.6,
                contain_thr: float = 0.8, max_n: int = 12) -> list[tuple[list[float], float]]:
    """Candidate food boxes from raw zero-shot detections (normalised xyxy).

    Drops low scores and near-whole-image boxes, removes duplicates (NMS), drops containers (a box holding two or
    more other boxes, e.g. a plate or tray), and returns at most max_n, ordered top-left to bottom-right.
    """
    order = sorted((i for i, s in enumerate(scores) if s >= min_score and area(boxes[i]) <= max_area),
                   key=lambda i: -scores[i])
    kept: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < iou_thr for j in kept):
            kept.append(i)
    containers = {a for a in kept
                  if sum(1 for b in kept if b != a and intersection(boxes[a], boxes[b]) >= contain_thr * area(boxes[b])) >= 2}
    kept = [i for i in kept if i not in containers][:max_n]
    kept.sort(key=lambda i: (round((boxes[i][1] + boxes[i][3]) / 2 * 4), (boxes[i][0] + boxes[i][2]) / 2))
    return [(list(boxes[i]), scores[i]) for i in kept]


_CELL = re.compile(r"^([A-J])(10|[1-9])$", re.I)


def grid_box(spec: str) -> list[float]:
    """'C3:E6' -> normalised [x1, y1, x2, y2]: from the top-left of cell C3 to the bottom-right of cell E6."""
    a, b = spec.split(":")
    ma, mb = _CELL.match(a.strip()), _CELL.match(b.strip())
    if not ma or not mb:
        raise ValueError(f"bad grid box {spec!r}")
    c1, r1 = ord(ma.group(1).upper()) - 65, int(ma.group(2)) - 1
    c2, r2 = ord(mb.group(1).upper()) - 65, int(mb.group(2)) - 1
    x1, x2 = sorted((c1, c2))
    y1, y2 = sorted((r1, r2))
    return [x1 / GRID, y1 / GRID, (x2 + 1) / GRID, (y2 + 1) / GRID]


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    keep: list[int] = []
    for i in np.argsort(-scores):
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(int(i))
    return keep
