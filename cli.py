"""
Konsol režimi — wideo faýl we janly kamera üçin.

Mysallar:

  # wideo faýl, sütünler awtomatiki, hemmesiniň arasy 10 m
  python cli.py --source samples/realistic.mp4 --distances 10 --show

  # bölekleriň arasy dürli (4 sütün → 3 aralyk)
  python cli.py --source road.mp4 --distances 12,15.5,12 --csv out.csv --video out.mp4

  # web-interfeýsden ýatda saklanan kalibrowka (sütünler + aralyklar)
  python cli.py --source road.mp4 --config calibration.json

  # 0-njy USB-kamera, janly (sütünler ilkinji kadrlar boýunça gözlenýär)
  python cli.py --source 0 --distances 10 --show

  # IP-kamera
  python cli.py --source rtsp://user:pass@192.168.1.10:554/stream --config calibration.json --show
"""

from __future__ import annotations

import argparse
import sys

from speedcam.color import color_name, label_name
from speedcam.detector import build_detector
from speedcam.pipeline import RunConfig, export_csv, export_json, run
from speedcam.poles import detect_poles
from speedcam.video import estimate_background, resolve_source

# Windows konsoly (cp1252) "→" we türkmen harplaryny çap edip bilmeýär — UTF-8-e geçirýäris
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(";", ",").split(",") if x.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SpeedCam — awtoulagyň sütünleriň arasyndaky tizligi")
    p.add_argument("--source", required=True, help="faýl, kamera indeksi (0,1,...) ýa-da RTSP/HTTP URL")
    p.add_argument("--config", help="web-interfeýsden calibration.json (sütünler + aralyklar)")
    p.add_argument("--poles", help="sütünleriň X ýerleşişi, otur bilen; berilmese — awtomatiki")
    p.add_argument("--distances", help="goňşy sütünleriň arasyndaky aralyklar, m (bir san — hemmesi üçin)")
    p.add_argument("--detector", choices=["yolo", "motion"], default="yolo")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--limit", type=float, default=None, help="tizlik çägi, km/sag")
    p.add_argument("--sensitivity", type=float, default=0.35, help="sütün gözlegiň çägi (0.1–0.9)")
    p.add_argument("--show", action="store_true", help="penjirede görkez (q / Esc — çykmak)")
    p.add_argument("--video", help="bellikli wideony ýatda sakla")
    p.add_argument("--csv", help="netijeleri CSV-e ýaz")
    p.add_argument("--json", help="netijeleri JSON-a ýaz")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-color", action="store_true", help="awtoulagyň reňkini kesgitleme")
    a = p.parse_args(argv)

    src = resolve_source(a.source)

    # --- sütünler we aralyklar ---
    if a.config:
        cfg = RunConfig.load(a.config)
        poles, distances = cfg.poles_x, cfg.distances_m
        if a.distances:
            distances = parse_floats(a.distances)
    else:
        if a.poles:
            poles = sorted(parse_floats(a.poles))
        else:
            print("Sütünler awtomatiki gözlenýär…", flush=True)
            bg = estimate_background(src)
            found = detect_poles(bg, rel_threshold=a.sensitivity)
            poles = [round(f.x, 1) for f in found]
            print(f"Tapylan sütün sany: {len(poles)} → X = {poles}")
        if not a.distances:
            print("--distances görkeziň (mysal: --distances 10 ýa-da --distances 12,15,12)", file=sys.stderr)
            return 2
        distances = parse_floats(a.distances)

    if len(poles) < 2:
        print("Azyndan 2 sütün gerek. --poles bilen el bilen beriň ýa-da --sensitivity peseldiň.", file=sys.stderr)
        return 2
    if len(distances) == 1 and len(poles) > 2:
        distances = distances * (len(poles) - 1)
    if len(distances) != len(poles) - 1:
        print(f"Sütün sany {len(poles)} → {len(poles) - 1} aralyk gerek, berlen: {len(distances)}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        poles_x=poles, distances_m=distances, speed_limit_kmh=a.limit,
        output_video=a.video, show_window=a.show, max_frames=a.max_frames,
        detect_color=not a.no_color,
    )
    print(f"Sütünler X: {poles}\nAralyklar, m: {distances}")

    det = build_detector(a.detector, model=a.model, conf=a.conf) if a.detector == "yolo" else build_detector("motion")

    def on_result(r):
        flag = "  !!! ÇÄKDEN GEÇDI" if a.limit and r.speed_kmh > a.limit else ""
        print(f"[{r.frame_to:.0f}] ID{r.track_id:<3} {label_name(r.label):<14} {color_name(r.color):<10} "
              f"P{r.pole_from}→P{r.pole_to} {r.direction}  {r.distance_m:g} m / {r.time_s:.2f} s  "
              f"= {r.speed_kmh:6.1f} km/sag{flag}", flush=True)

    def progress(i, n):
        print(f"\r  kadr {i}/{n}", end="", flush=True)

    res = run(src, det, cfg, progress=progress, on_result=on_result)
    print(f"\n{res.frames_processed} kadr {res.elapsed_s:.1f} sekuntda işlenildi")

    if res.summary:
        print("\nJemleme:")
        for s in res.summary:
            print(f"  ID{s['track_id']:<3} {label_name(s['label']):<14} {color_name(s['color']):<10} "
                  f"bölek: {s['segments']}  ortaça {s['avg_speed_kmh']} km/sag "
                  f"(iň pes {s['min_speed_kmh']}, iň ýokary {s['max_speed_kmh']})")
    else:
        print("Hiç bir awtoulag iki goňşy sütünden geçmedi.")

    if a.csv:
        export_csv(res, a.csv)
        print(f"CSV → {a.csv}")
    if a.json:
        export_json(res, a.json)
        print(f"JSON → {a.json}")
    if a.video:
        print(f"Wideo → {a.video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
