"""
Автоматическое нахождение столбов на кадре.

Столб при съёмке сбоку — это длинный почти вертикальный объект. Алгоритм:

1. Берём фон (медианный кадр, см. video.estimate_background) — на нём нет машин.
2. Canny → HoughLinesP: собираем отрезки, отклонение которых от вертикали
   не превышает max_tilt_deg и длина не меньше min_height_ratio * высота кадра.
3. Отрезки группируем по X (кластеры шириной cluster_px). Сила кластера =
   суммарная длина отрезков в нём. У столба две кромки (левая и правая), поэтому
   столбы дают очень «сильные» кластеры.
4. Кластеры слабее rel_threshold * max_strength отбрасываем, остальное — столбы.

Дополнительно: профиль вертикальных градиентов по столбцам (Sobel X) как
второй голос — помогает, если Hough «порвал» столб на куски.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Pole:
    x: float            # положение столба по X (пиксели)
    y_top: float        # верх видимой части
    y_bottom: float     # низ
    strength: float     # условная «сила» (суммарная длина кромок)

    def to_dict(self) -> dict:
        return asdict(self)


def _vertical_segments(gray: np.ndarray, min_len: int, max_tilt_deg: float):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=min_len, maxLineGap=max(4, min_len // 6),
    )
    segs = []
    if lines is None:
        return segs
    tan_max = math.tan(math.radians(max_tilt_deg))
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: (N,1,4), OpenCV 5: (N,4)
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        if dy == 0 or dx > tan_max * dy:
            continue
        segs.append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2), float(dy)))
    return segs


def detect_poles(
    image: np.ndarray,
    min_height_ratio: float = 0.18,
    max_tilt_deg: float = 8.0,
    cluster_ratio: float = 0.015,
    rel_threshold: float = 0.35,
    max_poles: int = 12,
    min_gap_ratio: float = 0.04,
) -> list[Pole]:
    """
    image            — кадр BGR (лучше фон без машин).
    min_height_ratio — минимальная высота столба как доля высоты кадра.
    max_tilt_deg     — допустимый наклон от вертикали.
    cluster_ratio    — ширина кластера по X как доля ширины кадра.
    rel_threshold    — порог силы относительно самого сильного кластера (0..1).
                       Меньше → больше кандидатов, больше → только явные столбы.
    min_gap_ratio    — два столба не могут быть ближе, чем эта доля ширины.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    min_len = max(10, int(min_height_ratio * h))
    segs = _vertical_segments(gray, min_len, max_tilt_deg)
    if not segs:
        return []

    # --- кластеризация по X ---
    segs.sort(key=lambda s: s[0])
    cluster_w = max(3.0, cluster_ratio * w)
    clusters: list[list[tuple]] = []
    for s in segs:
        if clusters and s[0] - clusters[-1][-1][0] <= cluster_w:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    cands: list[Pole] = []
    for cl in clusters:
        lens = np.array([s[3] for s in cl])
        xs = np.array([s[0] for s in cl])
        x = float(np.average(xs, weights=lens))
        cands.append(Pole(
            x=x,
            y_top=float(min(s[1] for s in cl)),
            y_bottom=float(max(s[2] for s in cl)),
            strength=float(lens.sum()),
        ))
    if not cands:
        return []

    # --- отсев слабых ---
    max_s = max(c.strength for c in cands)
    strong = [c for c in cands if c.strength >= rel_threshold * max_s]

    # --- non-maximum suppression по X: не ближе min_gap ---
    min_gap = min_gap_ratio * w
    strong.sort(key=lambda c: -c.strength)
    kept: list[Pole] = []
    for c in strong:
        if all(abs(c.x - k.x) >= min_gap for k in kept):
            kept.append(c)
        if len(kept) >= max_poles:
            break
    kept.sort(key=lambda c: c.x)
    return kept


def draw_poles(frame: np.ndarray, poles_x: list[float], labels: bool = True,
               color=(0, 200, 255)) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    for i, x in enumerate(poles_x):
        xi = int(round(x))
        cv2.line(out, (xi, 0), (xi, h), color, 2)
        if labels:
            cv2.rectangle(out, (xi + 4, 8), (xi + 48, 34), color, -1)
            cv2.putText(out, f"P{i + 1}", (xi + 8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out
