# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — Esasy.exe (Windows) / Esasy (Linux, macOS).

    pyinstaller --noconfirm Esasy.spec

Netije: dist/Esasy/Esasy.exe + _internal/ (ähli kitaphanalar, app.py, speedcam/, yolov8n.pt).
Tutuş dist/Esasy papkasyny göçürmeli (zip) — exe ýeke özi işlemeýär.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "cli.py"), "."),
    (str(ROOT / "speedcam"), "speedcam"),
]
for model in ("yolov8n.pt", "yolov8s.pt", "yolo11n.pt"):
    if (ROOT / model).is_file():
        datas.append((str(ROOT / model), "."))

binaries = []
hiddenimports = [
    "cli", "speedcam", "speedcam.color", "speedcam.detector", "speedcam.paths", "speedcam.pipeline",
    "speedcam.poles", "speedcam.speed", "speedcam.video",
    # app.py streamlit tarapyndan exec edilýär — onuň importlaryny aýdyň görkezýäris
    "pandas", "plotly", "plotly.express", "cv2", "numpy",
    "streamlit.web.bootstrap", "streamlit.runtime.scriptrunner.magic_funcs",
]

# Streamlit (static faýllar, konfig), ultralytics (cfg/*.yaml, tracker yaml), plotly (json şablonlary)
for pkg in ("streamlit", "ultralytics", "plotly", "lap", "narwhals"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:  # noqa: BLE001
        print(f"[spec] collect_all({pkg}) skipped: {e}")

# importlib.metadata.version(...) bilen barlanýan paketler
for pkg in ("streamlit", "ultralytics", "torch", "torchvision", "lap", "opencv-python", "numpy",
            "pandas", "plotly", "pywebview", "altair", "pyarrow", "tqdm", "pillow", "pyyaml",
            "requests", "scipy", "psutil", "matplotlib"):
    try:
        datas += copy_metadata(pkg)
    except Exception:  # noqa: BLE001
        pass

hiddenimports += collect_submodules("webview") if sys.platform.startswith("win") else []

excludes = [   # üns: "unittest" aýyrmaly däl — torch oňa daýanýar
    "tkinter", "pytest", "IPython", "jupyter", "notebook",
    "torch.utils.tensorboard", "tensorboard", "onnx", "onnxruntime", "wandb", "mlflow", "comet_ml",
]

a = Analysis(
    [str(ROOT / "Esasy.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Esasy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # konsol bar (CI-de --selftest/--cli çykyşy görünýär); penjire režiminde Esasy.py ony gizleýär
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "Esasy.ico") if (ROOT / "assets" / "Esasy.ico").is_file() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Esasy",
)
