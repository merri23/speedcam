"""Awtoulagyň reňkini ramkanyň merkezi bölegi boýunça kesgitlemek (HSV-ewristika)."""

from __future__ import annotations

import cv2
import numpy as np

# Türkmen dilinde reňk atlary (interfeýs üçin)
COLOR_NAMES = {
    "white": "ak", "silver": "kümüşsöw", "gray": "çal", "black": "gara",
    "red": "gyzyl", "orange": "mämişi", "yellow": "sary", "green": "ýaşyl",
    "cyan": "açyk gök", "blue": "gök", "purple": "melewşe", "unknown": "näbelli",
}
COLOR_NAMES_TK = COLOR_NAMES

# Rus dilinde (gerek bolsa)
COLOR_NAMES_RU = {
    "white": "белый", "silver": "серебристый", "gray": "серый", "black": "чёрный",
    "red": "красный", "orange": "оранжевый", "yellow": "жёлтый", "green": "зелёный",
    "cyan": "голубой", "blue": "синий", "purple": "фиолетовый", "unknown": "н/д",
}

# Obýekt görnüşleri (YOLO klaslary) türkmençe
LABELS = {
    "car": "awtoulag", "truck": "ýük awtoulagy", "bus": "awtobus",
    "motorcycle": "motosikl", "moving": "hereket edýän",
}


def color_name(key: str) -> str:
    return COLOR_NAMES.get(key, key)


def label_name(key: str) -> str:
    return LABELS.get(key, key)


def _hue_name(h: float) -> str:
    # OpenCV: H 0..179 aralygynda
    if h < 8 or h >= 168:
        return "red"
    if h < 20:
        return "orange"
    if h < 34:
        return "yellow"
    if h < 80:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    return "purple"


def estimate_color(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int]]:
    """
    (reňk ady, ortaça BGR) gaýtarýar.
    Ramkanyň merkezi bölegi (kuzow) alynýar; gyralary (fon, tigirler, aýnalar) taşlanýar.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return "unknown", (0, 0, 0)
    # merkezi zona: ini boýunça 20..80%, beýikligi boýunça 35..75% (aýnadan aşak, tigirden ýokary)
    cx1, cx2 = x1 + int(0.2 * bw), x1 + int(0.8 * bw)
    cy1, cy2 = y1 + int(0.35 * bh), y1 + int(0.75 * bh)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "unknown", (0, 0, 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hh, ss, vv = np.median(hsv, axis=0)
    mean_bgr = tuple(int(v) for v in crop.reshape(-1, 3).mean(axis=0))

    if vv < 50:
        name = "black"
    elif ss < 40:
        name = "white" if vv > 190 else ("silver" if vv > 120 else "gray")
    else:
        # hromatik reňkler üçin H medianasy diňe doýgun pikseller boýunça
        sat_px = hsv[hsv[:, 1] >= 40]
        hue = float(np.median(sat_px[:, 0])) if len(sat_px) else hh
        name = _hue_name(hue)
    return name, mean_bgr
