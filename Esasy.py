"""
Esasy — SpeedCam-yň başlangyç faýly (Esasy.exe şundan ýygnalýar).

Nähili işleýär:
  * Esasy.exe            → özüni «--server» baýdagy bilen ikinji proses hökmünde
                           işledýär (Streamlit serweri, diňe 127.0.0.1-de), port
                           açylýança garaşýar we aýratyn programma penjiresinde
                           (pywebview, Windows-da Edge WebView2) interfeýsi açýar.
                           Penjire ýapylanda serwer hem ýapylýar.
  * Esasy.exe --server   → diňe serwer (içki ulanyş, el bilen çagyrmak gerek däl).
  * Esasy.exe --cli ...  → konsol / kamera režimi (cli.py-yň argumentleri), mysal:
                           Esasy.exe --cli --source 0 --config calibration.json --show
  * Esasy.exe --selftest → kitaphanalaryň we modeliň exe-de bardygyny barlaýar.

Brauzer gerek däl, kitaphanalar gurnalmaýar — hemmesi exe-niň içinde.
Pywebview ýok bolsa (ýa-da WebView2 ýok bolsa) — adaty brauzerde açýar.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Windows konsoly (cp1252) "→" we türkmen harplaryny çap edip bilmeýär — UTF-8-e geçirýäris
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

APP_TITLE = "SpeedCam — awtoulagyň sütünleriň arasyndaky tizligi"
HOST = "127.0.0.1"


def _bundle_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _streamlit_env(port: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "SPEEDCAM_PORT": str(port),
        "YOLO_OFFLINE": "1",                 # ultralytics internete çykmasyn
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def _streamlit_options(port: int) -> dict:
    """bootstrap.run üçin konfigurasiýa (env-üsti bilen däl: ol diňe CLI-de işleýär)."""
    return {
        "server.headless": True,
        "server.address": HOST,
        "server.port": port,
        "server.fileWatcherType": "none",
        "server.enableXsrfProtection": False,
        "server.maxUploadSize": 2000,        # MB — uly wideolar üçin
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
        "client.toolbarMode": "minimal",
    }


# --------------------------------------------------------------------------- #
#  Serwer režimi (çaga proses)
# --------------------------------------------------------------------------- #
def run_server() -> None:
    bundle = _bundle_dir()
    app_path = bundle / "app.py"
    os.chdir(str(bundle))                 # app.py-yň import speedcam we model faýly tapylsyn
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))

    port = int(os.environ.get("SPEEDCAM_PORT") or 8501)
    from streamlit.web import bootstrap

    opts = _streamlit_options(port)
    bootstrap.load_config_options(opts)          # CLI muny özi edýär; biz bootstrap-y göni çagyrýarys
    bootstrap.run(str(app_path), False, [], opts)


# --------------------------------------------------------------------------- #
#  Esasy režim (penjire)
# --------------------------------------------------------------------------- #
def _wait_for_server(port: int, proc: subprocess.Popen, timeout: float = 120.0) -> bool:
    url = f"http://{HOST}:{port}/_stcore/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False


def _spawn_server(port: int) -> subprocess.Popen:
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--server"]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--server"]
    creation = 0
    if os.name == "nt":
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log = open(Path(_log_dir()) / "server.log", "a", encoding="utf-8")
    return subprocess.Popen(cmd, env=_streamlit_env(port), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=creation)


def _hide_own_console() -> None:
    """Exe konsol režiminde ýygnalýar (CI-de çykyş görünsin diýip); penjire režiminde konsoly gizleýäris."""
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()  # type: ignore[attr-defined]
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)       # SW_HIDE  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    d = Path(base) / "SpeedCam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _open_window(url: str, on_closed) -> None:
    """pywebview penjiresi; bolmasa — brauzer."""
    try:
        import webview  # pywebview

        window = webview.create_window(APP_TITLE, url, width=1400, height=900, min_size=(1000, 700))
        window.events.closed += on_closed
        gui = "edgechromium" if os.name == "nt" else None
        webview.start(gui=gui, private_mode=False)
    except Exception as e:  # noqa: BLE001  (webview ýok / WebView2 ýok)
        print(f"[Esasy] penjire açylmady ({e}); brauzer ulanylýar", flush=True)
        import webbrowser

        webbrowser.open(url)
        try:
            while True:            # serwer işleýärkä garaşýarys (Ctrl+C — çykmak)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        on_closed()


def selftest() -> int:
    """Esasy.exe --selftest: ähli kitaphanalar we model exe-niň içinde barmy (CI we ulanyjy üçin)."""
    bundle = _bundle_dir()
    os.chdir(str(bundle))
    sys.path.insert(0, str(bundle))
    import numpy as np

    import cv2, pandas, plotly.express, streamlit  # noqa: F401,E401
    from speedcam.detector import build_detector
    from speedcam.paths import resolve_model
    from speedcam.pipeline import RunConfig  # noqa: F401
    from speedcam.poles import detect_poles

    model_path = resolve_model("yolov8n.pt")
    print(f"[selftest] python {sys.version.split()[0]}  streamlit {streamlit.__version__}  cv2 {cv2.__version__}")
    print(f"[selftest] model: {model_path}  exists={Path(model_path).is_file()}")
    det = build_detector("yolo", model=model_path)
    frame = np.zeros((480, 640, 3), np.uint8)
    dets = det.track(frame)
    poles = detect_poles(frame)
    print(f"[selftest] YOLO ok ({len(dets)} det on blank frame), poles ok ({len(poles)})")
    print("[selftest] OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--cli" in sys.argv:            # Esasy.exe --cli --source 0 --config calibration.json --show
        bundle = _bundle_dir()
        os.chdir(str(bundle))
        sys.path.insert(0, str(bundle))
        import cli

        args = [x for x in sys.argv[1:] if x != "--cli"]
        return cli.main(args)
    if "--server" in sys.argv:
        run_server()
        return 0

    if os.name == "nt" and "--no-window" not in sys.argv:
        _hide_own_console()

    port = int(os.environ.get("SPEEDCAM_PORT") or _free_port())
    proc = _spawn_server(port)
    url = f"http://{HOST}:{port}"

    if not _wait_for_server(port, proc):
        proc.kill()
        msg = f"Serwer işe girmedi. Žurnal: {_log_dir() / 'server.log'}"
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "SpeedCam", 0x10)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            print(msg, file=sys.stderr)
        return 1

    def shutdown(*_):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):          # launcher öldürilse — serwer hem ýapylsyn
        try:
            signal.signal(sig, lambda *_: (shutdown(), sys.exit(0)))
        except (ValueError, OSError):
            pass

    if "--no-window" in sys.argv:      # CI / test: serweri işledip, penjiresiz garaşmak
        print(url, flush=True)
        try:
            proc.wait()
        finally:
            shutdown()
        return 0

    _open_window(url, shutdown)
    shutdown()
    return 0


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing
        multiprocessing.freeze_support()
    sys.exit(main())
