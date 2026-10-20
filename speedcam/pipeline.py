"""
Конвейер: источник → детектор/трекер → пересечение столбов → скорость → отчёт.

Используется и веб-интерфейсом (app.py), и консольным режимом (cli.py),
в том числе для живой камеры.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .color import color_name, estimate_color, label_name
from .detector import Detection, Detector
from .poles import draw_poles
from .speed import SegmentResult, SpeedEstimator
from .video import VideoInfo, iter_frames, make_writer, open_video


@dataclass
class RunConfig:
    poles_x: list[float]
    distances_m: list[float]
    speed_limit_kmh: Optional[float] = None
    output_video: Optional[str] = None
    show_window: bool = False
    max_frames: Optional[int] = None
    detect_color: bool = True

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "poles_x": self.poles_x,
            "distances_m": self.distances_m,
            "speed_limit_kmh": self.speed_limit_kmh,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str, **override) -> "RunConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.update(override)
        return cls(**d)


@dataclass
class RunResult:
    info: VideoInfo
    segments: list[SegmentResult]
    summary: list[dict]
    frames_processed: int
    elapsed_s: float
    output_video: Optional[str] = None
    snapshots: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (подпись, кадр)

    def rows(self) -> list[dict]:
        return [s.as_row() for s in self.segments]


# --------------------------------------------------------------------------- #
#  Отрисовка
# --------------------------------------------------------------------------- #
def _palette(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 7919)
    c = rng.integers(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def annotate(frame: np.ndarray, dets: list[Detection], est: SpeedEstimator,
             colors: dict[int, str], limit: Optional[float], frame_idx: int, fps: float) -> np.ndarray:
    out = draw_poles(frame, est.poles_x)
    for d in dets:
        x1, y1, x2, y2 = d.bbox_int()
        col = _palette(d.track_id)
        v = est.last_speed(d.track_id)
        over = limit is not None and v is not None and v > limit
        if over:
            col = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.circle(out, (int(d.cx), int(d.cy)), 4, col, -1)
        cname = colors.get(d.track_id, "")
        txt = f"ID{d.track_id} {label_name(d.label)}"
        if cname and cname != "unknown":
            txt += f" ({color_name(cname)})"
        if v is not None:
            txt += f"  {v:.0f} km/h" + ("  !!!" if over else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        # отметки пройденных столбов
        for i in est.crossed_poles(d.track_id):
            px = int(est.poles_x[i])
            cv2.circle(out, (px, int(d.cy)), 7, col, 2)
    cv2.putText(out, f"kadr {frame_idx}  t={frame_idx / fps:6.2f}s", (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
#  Основной цикл
# --------------------------------------------------------------------------- #
def run(
    source,
    detector: Detector,
    cfg: RunConfig,
    progress: Optional[Callable[[int, int], None]] = None,
    on_result: Optional[Callable[[SegmentResult], None]] = None,
    on_frame: Optional[Callable[[np.ndarray, int], None]] = None,
) -> RunResult:
    cap, info = open_video(source)
    est = SpeedEstimator(cfg.poles_x, cfg.distances_m, info.fps)
    colors: dict[int, str] = {}
    writer = None
    if cfg.output_video:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.output_video)), exist_ok=True)
        writer = make_writer(cfg.output_video, info.fps, info.width, info.height)

    snapshots: list[tuple[str, np.ndarray]] = []
    known_width: dict[int, float] = {}   # ширина машины по неразрезанным кадрам

    def stable_center(d: Detection) -> float:
        """
        У края кадра рамка обрезается и её центр «отстаёт» от настоящего центра
        машины. Если рамка упёрлась в край, восстанавливаем центр по ширине,
        измеренной, когда машина была целиком в кадре.
        """
        at_left, at_right = d.x1 <= 2, d.x2 >= info.width - 2
        if not at_left and not at_right:
            known_width[d.track_id] = max(known_width.get(d.track_id, 0.0), d.width)
            return d.cx
        w = known_width.get(d.track_id)
        if w is None or w <= d.width:
            return d.cx
        return d.x1 + w / 2 if at_right else d.x2 - w / 2

    t0 = time.time()
    n = 0
    total = info.frame_count if cfg.max_frames is None else min(info.frame_count, cfg.max_frames)
    try:
        for idx, frame in iter_frames(cap):
            if cfg.max_frames is not None and idx >= cfg.max_frames:
                break
            dets = detector.track(frame)
            new_results: list[SegmentResult] = []
            for d in dets:
                if cfg.detect_color and (d.track_id not in colors or idx % 15 == 0):
                    cname, _ = estimate_color(frame, d.bbox_int())
                    if cname != "unknown":
                        colors[d.track_id] = cname
                new_results += est.update(d.track_id, stable_center(d), idx, d.label, colors.get(d.track_id))

            need_draw = writer is not None or on_frame is not None or cfg.show_window or new_results
            vis = annotate(frame, dets, est, colors, cfg.speed_limit_kmh, idx, info.fps) if need_draw else None

            for r in new_results:
                if on_result:
                    on_result(r)
                if vis is not None and len(snapshots) < 12:
                    snapshots.append((f"ID{r.track_id} P{r.pole_from}→P{r.pole_to}: {r.speed_kmh:.1f} km/sag", vis.copy()))

            if writer is not None and vis is not None:
                writer.write(vis)
            if on_frame is not None and vis is not None:
                on_frame(vis, idx)
            if cfg.show_window and vis is not None:
                cv2.imshow("speedcam", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            n += 1
            if progress and total > 0 and (idx % 10 == 0 or idx == total - 1):
                progress(idx + 1, total)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.show_window:
            cv2.destroyAllWindows()

    return RunResult(
        info=info, segments=list(est.results), summary=est.track_summary(),
        frames_processed=n, elapsed_s=time.time() - t0,
        output_video=cfg.output_video, snapshots=snapshots,
    )


# --------------------------------------------------------------------------- #
#  Экспорт
# --------------------------------------------------------------------------- #
def export_csv(result: RunResult, path: str) -> None:
    import csv

    rows = result.rows()
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def export_json(result: RunResult, path: str) -> None:
    Path(path).write_text(json.dumps({
        "video": {"fps": result.info.fps, "width": result.info.width,
                  "height": result.info.height, "frames": result.frames_processed},
        "segments": result.rows(),
        "summary": result.summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def try_convert_h264(src: str) -> str:
    """Если есть ffmpeg — перекодируем в H.264, чтобы видео играло в браузере."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return src
    dst = str(Path(src).with_name(Path(src).stem + "_h264.mp4"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True, timeout=600,
        )
        return dst
    except Exception:
        return src
