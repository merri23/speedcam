"""
Faýl ýollary: adaty işletme (papkadan) we PyInstaller-bilen «doňdurylan» Esasy.exe üçin.

PyInstaller onedir gurluşynda goşulan faýllar (app.py, yolov8n.pt, speedcam/)
``sys._MEIPASS`` (dist/Esasy/_internal) papkasynda ýatýar.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Path:
    """Goşulan resurslaryň papkasy (frozen) ýa-da proýektiň köki."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """Esasy.exe-niň ýanyndaky papka (ulanyjynyň öz faýllary üçin) ýa-da proýektiň köki."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return bundle_dir()


def resolve_model(name: str) -> str:
    """
    YOLO modeliniň ýoly: 1) berlen ýol bar bolsa — şol; 2) exe-niň ýanynda;
    3) goşulan resurslarda; 4) bolmasa — at bolşy ýaly (ultralytics ýükläp alar).
    """
    p = Path(name)
    if p.is_file():
        return str(p)
    for d in (app_dir(), bundle_dir(), Path.cwd()):
        cand = d / p.name
        if cand.is_file():
            return str(cand)
    return name


def user_data_dir() -> Path:
    """Netijeler / wagtlaýyn faýllar üçin ýazyp bolýan papka."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "SpeedCam"
    d.mkdir(parents=True, exist_ok=True)
    return d
