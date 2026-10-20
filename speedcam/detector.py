"""
Детекторы машин с трекингом.

* YoloDetector  — нейросеть YOLO (ultralytics) + встроенный трекер ByteTrack.
                  Находит машины любого цвета; основной режим.
* MotionDetector — классическое вычитание фона (MOG2) + простой трекер по
                  центроидам. Не требует нейросети и интернета; запасной
                  режим и режим для тестов.

Оба отдают одинаковый список Detection, поэтому остальной код от них не зависит.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np

# Классы COCO, которые считаем «транспортом»
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    conf: float = 1.0
    label: str = "car"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def bbox_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class Detector(Protocol):
    name: str

    def track(self, frame: np.ndarray) -> list[Detection]: ...


# --------------------------------------------------------------------------- #
#  YOLO
# --------------------------------------------------------------------------- #
class YoloDetector:
    """
    model: 'yolov8n.pt' (самая быстрая, ~6 МБ), 'yolov8s.pt' (точнее), 'yolo11n.pt' ...
    Модель скачивается автоматически при первом запуске (нужен интернет один раз).
    """

    name = "yolo"

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.3,
        classes: Optional[list[int]] = None,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        import torch  # noqa: F401  (tertip möhüm: torch → torchvision → ultralytics; frozen exe üçin)
        import torchvision  # noqa: F401
        from ultralytics import YOLO  # импорт здесь, чтобы MotionDetector работал без torch

        from .paths import resolve_model

        self.model = YOLO(resolve_model(model))
        self.conf = conf
        self.classes = classes if classes is not None else list(VEHICLE_CLASSES)
        self.tracker = tracker
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker=self.tracker,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid, c, k in zip(xyxy, ids, confs, clss):
            out.append(
                Detection(
                    float(x1), float(y1), float(x2), float(y2),
                    track_id=int(tid), conf=float(c),
                    label=str(self.names.get(int(k), k)),
                )
            )
        return out


# --------------------------------------------------------------------------- #
#  Классический детектор движения + простой трекер
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    track_id: int
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    missed: int = 0
    age: int = 0


class MotionDetector:
    """
    Вычитание фона MOG2 → маска движения → контуры → bbox.
    Трекер: ближайший центроид (greedy), трек живёт max_missed кадров без детекций.
    """

    name = "motion"

    def __init__(
        self,
        min_area_ratio: float = 0.002,   # мин. площадь объекта как доля кадра
        max_missed: int = 10,
        max_match_dist_ratio: float = 0.15,  # макс. скачок центроида между кадрами (доля ширины)
        history: int = 200,
        var_threshold: float = 40.0,
        min_age_to_report: int = 2,
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area_ratio = min_area_ratio
        self.max_missed = max_missed
        self.max_match_dist_ratio = max_match_dist_ratio
        self.min_age_to_report = min_age_to_report
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def _detect_boxes(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        h, w = frame.shape[:2]
        mask = self.bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # убираем тени (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=3)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_ratio * w * h
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
        return boxes

    def track(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        boxes = self._detect_boxes(frame)
        max_dist = self.max_match_dist_ratio * w

        # сопоставление: жадно по минимальному расстоянию
        unmatched = list(range(len(boxes)))
        for t in self._tracks.values():
            t.age += 1
        pairs: list[tuple[float, int, int]] = []
        for tid, t in self._tracks.items():
            for j, (x1, y1, x2, y2) in enumerate(boxes):
                d = math.hypot((x1 + x2) / 2 - t.cx, (y1 + y2) / 2 - t.cy)
                if d <= max_dist:
                    pairs.append((d, tid, j))
        pairs.sort()
        used_t: set[int] = set()
        for d, tid, j in pairs:
            if tid in used_t or j not in unmatched:
                continue
            x1, y1, x2, y2 = boxes[j]
            t = self._tracks[tid]
            t.cx, t.cy, t.bbox, t.missed = (x1 + x2) / 2, (y1 + y2) / 2, boxes[j], 0
            used_t.add(tid)
            unmatched.remove(j)

        for j in unmatched:  # новые объекты → новые треки
            x1, y1, x2, y2 = boxes[j]
            self._tracks[self._next_id] = _Track(
                self._next_id, (x1 + x2) / 2, (y1 + y2) / 2, boxes[j]
            )
            used_t.add(self._next_id)
            self._next_id += 1

        for tid in list(self._tracks):  # не сопоставленные треки стареют и умирают
            t = self._tracks[tid]
            if tid not in used_t:
                t.missed += 1
            if t.missed > self.max_missed:
                del self._tracks[tid]

        out = []
        for t in self._tracks.values():
            if t.missed == 0 and t.age >= self.min_age_to_report:
                x1, y1, x2, y2 = t.bbox
                out.append(Detection(x1, y1, x2, y2, track_id=t.track_id, conf=1.0, label="moving"))
        return out


def build_detector(kind: str = "yolo", **kw) -> Detector:
    kind = kind.lower()
    if kind == "yolo":
        return YoloDetector(**kw)
    if kind == "motion":
        return MotionDetector(**kw)
    raise ValueError(f"Näbelli detektor: {kind}")
