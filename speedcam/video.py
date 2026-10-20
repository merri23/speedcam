"""Источник кадров: видеофайл, USB-камера (индекс) или IP-камера (RTSP/HTTP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np

Source = Union[str, int]


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int  # 0 / -1 для камер (неизвестно)

    @property
    def is_stream(self) -> bool:
        return self.frame_count <= 0


def resolve_source(src: Source) -> Source:
    """'0', '1' → индекс камеры; остальное — путь к файлу или URL."""
    if isinstance(src, int):
        return src
    s = str(src).strip()
    if s.isdigit():
        return int(s)
    return s


def open_video(src: Source) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(resolve_source(src))
    if not cap.isOpened():
        raise RuntimeError(f"Wideo çeşmesi açylmady: {src!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1 or fps > 240:  # камеры часто отдают 0 или мусор
        fps = 30.0
    info = VideoInfo(
        fps=float(fps),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    return cap, info


def iter_frames(cap: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    """Генератор (номер_кадра, кадр BGR)."""
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield idx, frame
        idx += 1


def read_first_frame(src: Source) -> tuple[np.ndarray, VideoInfo]:
    cap, info = open_video(src)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Wideo boş — ilkinji kadr okalmady")
        return frame, info
    finally:
        cap.release()


def estimate_background(src: Source, n_samples: int = 25) -> np.ndarray:
    """
    Оценка фона (кадр без машины): медиана по N равномерно выбранным кадрам.
    Движущиеся объекты «исчезают», а неподвижные столбы остаются чёткими —
    именно на этом кадре удобно искать столбы автоматически.
    Для камер (поток) берём N подряд идущих кадров.
    """
    cap, info = open_video(src)
    frames: list[np.ndarray] = []
    try:
        if info.is_stream:
            for _ in range(n_samples):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
        else:
            positions = np.linspace(0, max(info.frame_count - 1, 0), n_samples).astype(int)
            for p in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
                ok, f = cap.read()
                if ok:
                    frames.append(f)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("Fony kesgitlemek üçin kadrlar okalmady")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def make_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """VideoWriter (mp4v — работает везде; для браузера потом перекодируем ffmpeg'ом, см. pipeline.try_convert_h264)."""
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not w.isOpened():
        raise RuntimeError("VideoWriter döredilmedi")
    return w
