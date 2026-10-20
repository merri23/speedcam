"""
Расчёт скорости по пересечениям линий столбов.

Для каждого трека (машины) храним последнее положение центра по X и номер кадра.
Когда центр переходит через линию столба, момент пересечения вычисляем с
субкадровой точностью линейной интерполяцией:

    t = f_prev + (x_pole - x_prev) / (x_cur - x_prev) * (f_cur - f_prev)

Скорость на участке между столбами i и i+1:

    v = distance_m[i] / (|t_{i+1} - t_i| / fps)      [м/с]  → ×3.6 = км/ч
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

MS_TO_KMH = 3.6


@dataclass
class SegmentResult:
    track_id: int
    segment_index: int          # 0 → участок P1→P2
    pole_from: int              # 1-based
    pole_to: int                # 1-based
    frame_from: float
    frame_to: float
    distance_m: float
    time_s: float
    speed_kmh: float
    direction: str              # "→" или "←"
    label: str = "car"
    color: str = "unknown"

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh / MS_TO_KMH

    def as_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "color": self.color,
            "segment": f"P{self.pole_from}→P{self.pole_to}",
            "direction": self.direction,
            "frame_from": round(self.frame_from, 2),
            "frame_to": round(self.frame_to, 2),
            "time_s": round(self.time_s, 3),
            "distance_m": self.distance_m,
            "speed_ms": round(self.speed_ms, 2),
            "speed_kmh": round(self.speed_kmh, 1),
        }


@dataclass
class _TrackState:
    last_cx: Optional[float] = None
    last_frame: Optional[int] = None
    crossings: dict[int, float] = field(default_factory=dict)   # pole_idx → frame (float)
    reported: set[int] = field(default_factory=set)             # segment indices
    direction: str = "?"
    label: str = "car"
    color: str = "unknown"
    last_seen: int = 0


class SpeedEstimator:
    def __init__(self, poles_x: list[float], distances_m: list[float], fps: float):
        if len(poles_x) < 2:
            raise ValueError("Azyndan 2 sütün gerek")
        if len(distances_m) != len(poles_x) - 1:
            raise ValueError(
                f"Aralyk sany {len(poles_x) - 1} bolmaly (sütün sany {len(poles_x)}), "
                f"berlen: {len(distances_m)}"
            )
        order = sorted(range(len(poles_x)), key=lambda i: poles_x[i])
        if order != list(range(len(poles_x))):
            raise ValueError("Sütünler X boýunça çepden saga tertiplenen bolmaly")
        self.poles_x = [float(x) for x in poles_x]
        self.distances_m = [float(d) for d in distances_m]
        self.fps = float(fps)
        self._tracks: dict[int, _TrackState] = {}
        self.results: list[SegmentResult] = []

    # ------------------------------------------------------------------ #
    def update(self, track_id: int, cx: float, frame_idx: int,
               label: str = "car", color: Optional[str] = None) -> list[SegmentResult]:
        """Обновить положение трека. Возвращает новые завершённые участки."""
        st = self._tracks.setdefault(track_id, _TrackState())
        st.label = label
        st.last_seen = frame_idx
        if color and color != "unknown":
            st.color = color
        new: list[SegmentResult] = []

        if st.last_cx is not None and st.last_frame is not None and frame_idx > st.last_frame:
            dx = cx - st.last_cx
            if abs(dx) > 1e-6:
                st.direction = "→" if dx > 0 else "←"
            for i, px in enumerate(self.poles_x):
                crossed = (st.last_cx - px) * (cx - px) <= 0 and st.last_cx != cx
                if crossed and i not in st.crossings:
                    frac = (px - st.last_cx) / (cx - st.last_cx)
                    t = st.last_frame + frac * (frame_idx - st.last_frame)
                    st.crossings[i] = t
            new = self._emit(track_id, st)

        st.last_cx, st.last_frame = cx, frame_idx
        return new

    def _emit(self, track_id: int, st: _TrackState) -> list[SegmentResult]:
        out = []
        for i in range(len(self.poles_x) - 1):
            if i in st.reported or i not in st.crossings or (i + 1) not in st.crossings:
                continue
            t1, t2 = st.crossings[i], st.crossings[i + 1]
            dt = abs(t2 - t1) / self.fps
            if dt <= 0:
                continue
            v = self.distances_m[i] / dt * MS_TO_KMH
            res = SegmentResult(
                track_id=track_id, segment_index=i, pole_from=i + 1, pole_to=i + 2,
                frame_from=min(t1, t2), frame_to=max(t1, t2),
                distance_m=self.distances_m[i], time_s=dt, speed_kmh=v,
                direction=st.direction, label=st.label, color=st.color,
            )
            st.reported.add(i)
            self.results.append(res)
            out.append(res)
        return out

    # ------------------------------------------------------------------ #
    def track_summary(self) -> list[dict]:
        """Сводка по каждой машине: средняя скорость по всем пройденным участкам."""
        by_track: dict[int, list[SegmentResult]] = {}
        for r in self.results:
            by_track.setdefault(r.track_id, []).append(r)
        rows = []
        for tid, rs in sorted(by_track.items()):
            rs.sort(key=lambda r: r.segment_index)
            dist = sum(r.distance_m for r in rs)
            t = sum(r.time_s for r in rs)
            rows.append({
                "track_id": tid,
                "label": rs[0].label,
                "color": rs[-1].color,
                "direction": rs[0].direction,
                "segments": len(rs),
                "total_distance_m": round(dist, 2),
                "total_time_s": round(t, 3),
                "avg_speed_kmh": round(dist / t * MS_TO_KMH, 1) if t > 0 else None,
                "min_speed_kmh": round(min(r.speed_kmh for r in rs), 1),
                "max_speed_kmh": round(max(r.speed_kmh for r in rs), 1),
            })
        return rows

    def last_speed(self, track_id: int) -> Optional[float]:
        for r in reversed(self.results):
            if r.track_id == track_id:
                return r.speed_kmh
        return None

    def crossed_poles(self, track_id: int) -> set[int]:
        st = self._tracks.get(track_id)
        return set(st.crossings) if st else set()
