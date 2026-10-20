"""
Prototipiň web-interfeýsi (Streamlit) — türkmen dilinde.

Işe girizmek:  streamlit run app.py

Ädimler:
 1. Wideo ýükläň → programma sütünleri özi tapýar (düzedip bolýar).
 2. Sütünleriň arasyndaky aralygy giriziň (her gezek öz bahasy — şäher / ýol).
 3. Hasaplamany işlediň → tizlik tablisasy, grafik, bellikli wideo, CSV/JSON.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from speedcam.color import color_name, label_name
from speedcam.detector import build_detector
from speedcam.pipeline import RunConfig, export_csv, export_json, run, try_convert_h264
from speedcam.poles import detect_poles, draw_poles
from speedcam.video import estimate_background, read_first_frame

st.set_page_config(page_title="SpeedCam — sütünler boýunça tizlik", page_icon="🚗", layout="wide")

WORK_DIR = Path(tempfile.gettempdir()) / "speedcam_ui"  # wagtlaýyn faýllar (ýüklenen wideo, netijeler)
WORK_DIR.mkdir(exist_ok=True)


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- #
#  Gapdal panel — sazlamalar
# ----------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Sazlamalar")
    det_kind = st.radio("Detektor", ["YOLO (neýron tory)", "Hereket (OpenCV, neýron torsuz)"], index=0)
    det_kind = "yolo" if det_kind.startswith("YOLO") else "motion"
    model_name = st.selectbox("YOLO modeli", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"], index=0,
                              disabled=det_kind != "yolo",
                              help="n — iň çalt, s — has takyk. Ilkinji işletmede awtomatiki ýüklenýär.")
    conf = st.slider("Ynam çägi", 0.1, 0.9, 0.3, 0.05, disabled=det_kind != "yolo")
    speed_limit = st.number_input("Tizlik çägi, km/sag (0 — barlanmasyn)", 0.0, 300.0, 60.0, 5.0)
    detect_color = st.checkbox("Awtoulagyň reňkini kesgitle", True)
    save_video = st.checkbox("Bellikli wideony ýatda sakla", True)
    max_frames = st.number_input("Kadr sanyny çäklendir (0 — tutuş wideo)", 0, 100000, 0, 100)
    st.divider()
    st.subheader("Sütünleri gözlemek")
    sens = st.slider("Duýgurlyk", 0.1, 0.9, 0.35, 0.05,
                     help="Kiçi — köp kandidat tapýar; uly — diňe iň aýdyň sütünler.")
    min_h = st.slider("Sütüniň iň kiçi beýikligi, kadryň %-i", 5, 60, 18, 1)
    max_tilt = st.slider("Rugsat berilýän gyşarma, °", 2, 20, 8, 1)

st.title("🚗 SpeedCam — awtoulagyň sütünleriň arasyndaky tizligi")
st.caption("Prototip: häzir wideo faýl → soňra şol logika janly kamerada (cli.py-a serediň).")

# ----------------------------------------------------------------------------- #
#  1-nji ädim. Wideo
# ----------------------------------------------------------------------------- #
st.header("1️⃣ Wideo")
uploaded = st.file_uploader("Wideo ýükläň (mp4 / mov / avi / mkv)", type=["mp4", "mov", "avi", "mkv", "m4v"])

if uploaded is None:
    st.info("**Gapdaldan, hereketsiz kamera** bilen düşürilen wideo ýükläň: awtoulag çepden saga ýa-da "
            "sagdan çepe geçýär, sütünler dik obýekt hökmünde görünýär.")
    st.stop()

# bir gezek wagtlaýyn faýla ýazýarys
sig = (uploaded.name, uploaded.size)
if st.session_state.get("video_sig") != sig:
    vpath = WORK_DIR / f"input_{int(time.time())}_{uploaded.name}"
    vpath.write_bytes(uploaded.getbuffer())
    st.session_state.update(video_sig=sig, video_path=str(vpath), poles=None, result=None)
    for k in [k for k in st.session_state if k.startswith("dist_")]:
        del st.session_state[k]

video_path: str = st.session_state["video_path"]
first_frame, info = read_first_frame(video_path)

c1, c2, c3, c4 = st.columns(4)
c1.metric("FPS", f"{info.fps:.2f}")
c2.metric("Ölçegi", f"{info.width}×{info.height}")
c3.metric("Kadr sany", info.frame_count)
c4.metric("Dowamlylygy", f"{info.frame_count / info.fps:.1f} s")

# ----------------------------------------------------------------------------- #
#  2-nji ädim. Sütünler (awtomatiki + el bilen düzetmek)
# ----------------------------------------------------------------------------- #
st.header("2️⃣ Sütünler")


@st.cache_data(show_spinner="Awtoulagsyz fon gurulýar we sütünler gözlenýär…")
def _auto_poles(path: str, sens: float, min_h: int, max_tilt: int):
    bg = estimate_background(path)
    poles = detect_poles(bg, min_height_ratio=min_h / 100, max_tilt_deg=max_tilt, rel_threshold=sens)
    return bg, [round(p.x, 1) for p in poles]


bg, auto_xs = _auto_poles(video_path, sens, min_h, max_tilt)

if st.session_state.get("poles") is None or st.session_state.get("poles_auto") != auto_xs:
    st.session_state["poles"] = list(auto_xs)
    st.session_state["poles_auto"] = list(auto_xs)

colA, colB = st.columns([3, 2])
with colB:
    st.write(f"Awtomatiki tapyldy: **{len(auto_xs)}** sütün")
    st.caption("Artykmaç tapan bolsa ýa-da hemmesini tapmadyk bolsa — çepdäki duýgurlygy üýtgediň "
               "ýa-da X sanawyny düzediň (piksel, çepden saga, otur bilen).")
    poles_txt = st.text_input("Sütünleriň X boýunça ýerleşişi", value=", ".join(str(x) for x in st.session_state["poles"]))
    try:
        poles = sorted(float(x) for x in poles_txt.replace(";", ",").split(",") if x.strip())
    except ValueError:
        st.error("Sanaw okalmady — diňe sanlar, otur bilen aýrylan.")
        st.stop()
    st.session_state["poles"] = poles
    if st.button("↺ Awtomatiki bahalary gaýtar"):
        st.session_state["poles"] = list(auto_xs)
        st.rerun()
    show_bg = st.toggle("Fony görkez (awtoulagsyz)", False)

with colA:
    base = bg if show_bg else first_frame
    st.image(bgr2rgb(draw_poles(base, poles)), use_container_width=True,
             caption="Sary çyzyklar — sütünler P1…Pn (çepden saga)")

if len(poles) < 2:
    st.warning("Azyndan 2 sütün gerek. Duýgurlygy peseldiň ýa-da ýerleşişi el bilen giriziň.")
    st.stop()

# ----------------------------------------------------------------------------- #
#  3-nji ädim. Aralyklar
# ----------------------------------------------------------------------------- #
st.header("3️⃣ Sütünleriň arasyndaky aralyk")
st.caption("Her gezek girizilýär — şäherde we şäheriň daşynda aralyklar dürli bolýar.")

n_seg = len(poles) - 1
cc1, cc2 = st.columns([2, 1])
with cc1:
    same = st.number_input("Ähli bölekler üçin deň aralyk, m", 0.1, 10000.0, 10.0, 0.5)
with cc2:
    st.write("")
    st.write("")
    if st.button("Ähli böleklere ulan"):
        for i in range(n_seg):
            st.session_state[f"dist_{i}"] = float(same)

cols = st.columns(min(n_seg, 6))
distances: list[float] = []
for i in range(n_seg):
    key = f"dist_{i}"
    if key not in st.session_state:
        st.session_state[key] = float(same)
    with cols[i % len(cols)]:
        distances.append(st.number_input(f"P{i + 1} → P{i + 2}, m", min_value=0.1, max_value=10000.0,
                                         step=0.5, key=key))

# ----------------------------------------------------------------------------- #
#  4-nji ädim. Hasaplama
# ----------------------------------------------------------------------------- #
st.header("4️⃣ Hasaplama")
cfg = RunConfig(
    poles_x=poles, distances_m=distances,
    speed_limit_kmh=speed_limit or None,
    output_video=str(WORK_DIR / "annotated.mp4") if save_video else None,
    max_frames=int(max_frames) or None,
    detect_color=detect_color,
)

cfg_path = WORK_DIR / "calibration.json"
cfg.save(str(cfg_path))
st.download_button("💾 Kalibrowkany ýükle (kamera / cli.py üçin)", cfg_path.read_bytes(),
                   "calibration.json", "application/json")

if st.button("▶️ Tizligi hasapla", type="primary"):
    try:
        with st.spinner("Detektor ýüklenýär…"):
            det = build_detector(det_kind, model=model_name, conf=conf) if det_kind == "yolo" else build_detector("motion")
    except Exception as e:  # noqa: BLE001
        st.error(f"Detektor ýüklenmedi: {e}")
        st.stop()

    bar = st.progress(0.0, "Işlenilýär…")
    live = st.empty()
    log = st.container()

    def _progress(i, total):
        bar.progress(min(i / total, 1.0), f"Kadr {i} / {total}")

    def _on_result(r):
        log.write(f"🚗 ID{r.track_id} ({color_name(r.color)}, {label_name(r.label)}) "
                  f"P{r.pole_from}→P{r.pole_to}: **{r.speed_kmh:.1f} km/sag** "
                  f"({r.distance_m} m — {r.time_s:.2f} s)")

    last_shown = {"t": 0.0}

    def _on_frame(vis, idx):
        if time.time() - last_shown["t"] > 0.4:
            live.image(bgr2rgb(vis), use_container_width=True)
            last_shown["t"] = time.time()

    result = run(video_path, det, cfg, progress=_progress, on_result=_on_result, on_frame=_on_frame)
    bar.progress(1.0, f"Taýýar: {result.frames_processed} kadr, {result.elapsed_s:.1f} s "
                      f"({result.frames_processed / max(result.elapsed_s, 1e-6):.1f} FPS)")
    st.session_state["result"] = result

# ----------------------------------------------------------------------------- #
#  Netijeler
# ----------------------------------------------------------------------------- #
result = st.session_state.get("result")
if result is None:
    st.stop()

st.header("📊 Netijeler")
if not result.segments:
    st.warning("Hiç bir awtoulag iki goňşy sütünden geçmedi. Sütün çyzyklarynyň awtoulagyň geçýän ýerinde "
               "durýandygyny we detektoryň ony görýändigini barlaň (başga çäk / model synap görüň). "
               "Üns beriň: synthetic.mp4-däki çyzylan «awtoulagy» YOLO tanamaýar — onuň üçin "
               "«Hereket» detektoryny ýa-da realistic.mp4 faýlyny ulanyň.")
    st.stop()

COLS_SEG = {
    "track_id": "ID", "label": "Görnüşi", "color_tk": "Reňki", "segment": "Bölek", "direction": "Ugry",
    "frame_from": "Kadr (başy)", "frame_to": "Kadr (soňy)", "time_s": "Wagt, s",
    "distance_m": "Aralyk, m", "speed_ms": "Tizlik, m/s", "speed_kmh": "Tizlik, km/sag",
}
COLS_SUM = {
    "track_id": "ID", "label": "Görnüşi", "color_tk": "Reňki", "direction": "Ugry", "segments": "Bölek sany",
    "total_distance_m": "Jemi aralyk, m", "total_time_s": "Jemi wagt, s",
    "avg_speed_kmh": "Ortaça, km/sag", "min_speed_kmh": "Iň pes, km/sag", "max_speed_kmh": "Iň ýokary, km/sag",
}

df = pd.DataFrame(result.rows())
df["color_tk"] = df["color"].map(color_name)
df["label"] = df["label"].map(label_name)
summary = pd.DataFrame(result.summary)
summary["color_tk"] = summary["color"].map(color_name)
summary["label"] = summary["label"].map(label_name)

m1, m2, m3 = st.columns(3)
m1.metric("Awtoulag sany", len(summary))
m2.metric("Ortaça tizlik", f"{summary['avg_speed_kmh'].mean():.1f} km/sag")
m3.metric("Iň ýokary tizlik", f"{df['speed_kmh'].max():.1f} km/sag")

tab1, tab2, tab3, tab4 = st.tabs(["Awtoulaglar boýunça", "Bölekler boýunça", "Grafik", "Wideo we kadrlar"])
with tab1:
    st.dataframe(summary[list(COLS_SUM)].rename(columns=COLS_SUM), use_container_width=True, hide_index=True)
with tab2:
    show = df[list(COLS_SEG)].rename(columns=COLS_SEG)
    kmh = COLS_SEG["speed_kmh"]
    if speed_limit:
        st.dataframe(show.style.apply(
            lambda r: ["background-color: #ffd6d6" if r[kmh] > speed_limit else "" for _ in r], axis=1),
            use_container_width=True, hide_index=True)
    else:
        st.dataframe(show, use_container_width=True, hide_index=True)
with tab3:
    try:
        import plotly.express as px

        fig = px.bar(df, x="segment", y="speed_kmh", color=df["track_id"].astype(str), barmode="group",
                     labels={"segment": "Bölek", "speed_kmh": "km/sag", "color": "Awtoulag (ID)"},
                     title="Her bölekdäki tizlik")
        if speed_limit:
            fig.add_hline(y=speed_limit, line_dash="dash", line_color="red", annotation_text="çäk")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.pivot_table(index="segment", columns="track_id", values="speed_kmh"))
with tab4:
    if result.output_video and os.path.exists(result.output_video):
        playable = try_convert_h264(result.output_video)
        st.video(playable)
        st.download_button("⬇️ Bellikli wideony ýükle", open(playable, "rb").read(),
                           "annotated.mp4", "video/mp4")
    if result.snapshots:
        st.subheader("Bellige alnan pursatlar")
        sc = st.columns(3)
        for i, (cap_txt, img) in enumerate(result.snapshots):
            sc[i % 3].image(bgr2rgb(img), caption=cap_txt, use_container_width=True)

st.subheader("Eksport")
e1, e2 = st.columns(2)
csv_p, json_p = WORK_DIR / "results.csv", WORK_DIR / "results.json"
export_csv(result, str(csv_p))
export_json(result, str(json_p))
e1.download_button("⬇️ CSV", csv_p.read_bytes(), "speed_results.csv", "text/csv")
e2.download_button("⬇️ JSON", json_p.read_bytes(), "speed_results.json", "application/json")
