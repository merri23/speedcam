"""
Генератор синтетического тест-видео: дорога, 4 столба, машина едет слева направо
с РАЗНОЙ скоростью на каждом участке. Истинные скорости печатаются в консоль —
по ним проверяем точность программы (режим --detector motion).

    python samples/make_synthetic.py            → samples/synthetic.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import sys

import cv2
import numpy as np

# Windows konsoly (cp1252) "→" we türkmen harplaryny çap edip bilmeýär — UTF-8-e geçirýäris
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

W, H, FPS = 1280, 720, 30.0
POLES_X = [220, 500, 780, 1060]          # пиксели
DIST_M = 10.0                             # метров между соседними столбами (одинаково)
M_PER_PX = DIST_M / (POLES_X[1] - POLES_X[0])
SEG_SPEEDS_KMH = [36.0, 54.0, 72.0]       # истинная скорость на участках P1→P2, P2→P3, P3→P4
CAR_W, CAR_H = 170, 70


def kmh_to_px_per_frame(v: float) -> float:
    return v / 3.6 / M_PER_PX / FPS


def speed_at(x: float) -> float:
    """Скорость (км/ч) в зависимости от положения центра машины."""
    if x < POLES_X[1]:
        return SEG_SPEEDS_KMH[0]
    if x < POLES_X[2]:
        return SEG_SPEEDS_KMH[1]
    return SEG_SPEEDS_KMH[2]


def load_sprite(path: str, width: int = 460) -> np.ndarray:
    """Фото настоящего транспорта (samples/real_vehicle.jpg) → спрайт для «реалистичного» теста YOLO."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if (w, h) == (810, 1080):            # real_vehicle.jpg: вырезаем сам автобус
        img = img[225:720, 10:800]
    scale = width / img.shape[1]
    return cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def draw_scene(car_cx: float, color_bgr, rng: np.random.Generator, sprite: np.ndarray | None = None) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (190, 170, 140)                       # небо (BGR)
    cv2.rectangle(img, (0, 380), (W, H), (70, 70, 70), -1)        # асфальт
    cv2.rectangle(img, (0, 360), (W, 380), (60, 120, 60), -1)     # обочина
    for x in range(0, W, 120):                                    # разметка
        cv2.rectangle(img, (x, 540), (x + 60, 548), (230, 230, 230), -1)
    # деревья-помехи (не вертикальные, чтобы не путать со столбами)
    for tx in (130, 640, 1180):
        cv2.circle(img, (tx, 300), 55, (40, 110, 40), -1)
    # столбы
    for px in POLES_X:
        cv2.rectangle(img, (px - 7, 120), (px + 7, 420), (90, 90, 95), -1)
        cv2.rectangle(img, (px - 7, 120), (px + 7, 420), (40, 40, 40), 1)
        cv2.rectangle(img, (px - 20, 115), (px + 20, 128), (60, 60, 60), -1)  # перекладина
    # машина
    cx, cy = int(car_cx), 470
    if sprite is not None:                       # фото настоящего транспорта
        sh, sw = sprite.shape[:2]
        x1, y1 = cx - sw // 2, 560 - sh          # колёса на уровне дороги
        sx1, sy1 = max(0, -x1), max(0, -y1)
        sx2, sy2 = min(sw, W - x1), min(sh, H - y1)
        if sx2 > sx1 and sy2 > sy1:
            img[y1 + sy1:y1 + sy2, x1 + sx1:x1 + sx2] = sprite[sy1:sy2, sx1:sx2]
        noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x1, y1 = cx - CAR_W // 2, cy - CAR_H // 2
    cv2.rectangle(img, (x1, y1), (x1 + CAR_W, y1 + CAR_H), color_bgr, -1)
    cv2.rectangle(img, (x1 + 35, y1 - 32), (x1 + CAR_W - 35, y1), color_bgr, -1)       # кабина
    cv2.rectangle(img, (x1 + 42, y1 - 26), (x1 + CAR_W - 42, y1 - 4), (200, 220, 230), -1)  # стекло
    for wx in (x1 + 35, x1 + CAR_W - 35):
        cv2.circle(img, (wx, y1 + CAR_H), 20, (20, 20, 20), -1)
        cv2.circle(img, (wx, y1 + CAR_H), 9, (160, 160, 160), -1)
    # лёгкий шум как у реальной камеры
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("synthetic.mp4")))
    ap.add_argument("--color", default="red", choices=["red", "blue", "white", "black", "yellow"])
    ap.add_argument("--cars", type=int, default=1, help="yzly-yzyna näçe awtoulag geçmeli")
    ap.add_argument("--sprite", help="çyzgynyň ýerine hakyky ulagyň suraty (YOLO testi üçin), "
                                     "mysal: samples/real_vehicle.jpg")
    a = ap.parse_args()
    sprite = load_sprite(a.sprite) if a.sprite else None

    colors = {"red": (30, 30, 220), "blue": (200, 80, 30), "white": (245, 245, 245),
              "black": (25, 25, 25), "yellow": (30, 220, 240)}
    order = ["red", "blue", "white", "yellow", "black"]
    rng = np.random.default_rng(0)
    w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # 1 секунда пустой дороги в начале — чтобы детектор движения «выучил» фон
    for _ in range(int(FPS)):
        w.write(draw_scene(-1000, (0, 0, 0), rng, None))

    for k in range(a.cars):
        col = colors[a.color] if a.cars == 1 else colors[order[k % len(order)]]
        x = -(sprite.shape[1] if sprite is not None else CAR_W)
        while x < W + (sprite.shape[1] if sprite is not None else CAR_W):
            w.write(draw_scene(x, col, rng, sprite))
            x += kmh_to_px_per_frame(speed_at(x))
        for _ in range(int(FPS / 2)):
            w.write(draw_scene(-1000, (0, 0, 0), rng, None))
    w.release()

    print(f"Wideo: {a.out}")
    print(f"FPS {FPS}, sütünler X = {POLES_X}, aralyk {DIST_M} m")
    print("Bölekler boýunça hakyky tizlikler:",
          ", ".join(f"P{i + 1}→P{i + 2} = {v} km/sag" for i, v in enumerate(SEG_SPEEDS_KMH)))
    print(f"Barlag:  python cli.py --source {a.out} --distances {DIST_M} --detector {'yolo' if sprite is not None else 'motion'}")


if __name__ == "__main__":
    main()
