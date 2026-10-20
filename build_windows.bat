@echo off
rem ============================================================
rem  Esasy.exe-ni Windows-da el bilen ýygnamak (GitHub Actions-syz).
rem  Talap: Python 3.10-3.12 gurnalan we PATH-da.
rem  Netije: dist\Esasy\Esasy.exe  (tutuş dist\Esasy papkasyny göçüriň)
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist .venv-build (
    python -m venv .venv-build || goto :err
)
call .venv-build\Scripts\activate.bat

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :err
pip install -r requirements.txt pyinstaller pywebview || goto :err

if not exist yolov8n.pt (
    echo YOLO modeli ýüklenýär...
    curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt || goto :err
)

pyinstaller --noconfirm --clean Esasy.spec || goto :err

echo.
echo  TAÝÝAR:  dist\Esasy\Esasy.exe
echo  Paýlamak üçin dist\Esasy papkasyny zip ediň.
goto :eof

:err
echo.
echo  ÝALŇYŞLYK — ýokardaky habarlara serediň.
exit /b 1
