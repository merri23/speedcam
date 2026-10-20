# SpeedCam — awtoulagyň sütünleriň arasyndaky tizligi (prototip)

Hereketsiz kamera bilen gapdaldan düşürilen wideo boýunça **islendik reňkdäki**
awtoulagy tapýan, **sütünleri awtomatiki** kesgitleýän we goňşy sütünleriň
arasyndaky her bölekde tizligi hasaplaýan Python-programma. Sütünleriň
arasyndaky aralyk her gezek girizilýär (şäherde we şäheriň daşynda dürli bolýar).

Prototip wideo faýl bilen işleýär; şol logika hiç üýtgetmesiz USB- ýa-da
IP-kamerada işledilýär (`cli.py --source 0` / `--source rtsp://…`).

## Nähili işleýär

```
wideo ──► YOLO (awtoulagy tapmak) ──► ByteTrack (kadrdan kadra awtoulagyň ID-si)
                                                      │
fon (kadrlaryň medianasy) ──► sütün gözlegi ──► P1…Pn çyzyklary   │
                                                      ▼
     awtoulagyň merkeziniň Pi çyzygyndan geçen pursaty (kadryň bölegine çenli takyk)
                                                      ▼
            v = aralyk(Pi, Pi+1) / ((t_{i+1} − t_i) / FPS)  →  km/sag
```

* **Islendik reňkdäki awtoulag** — YOLO «car/truck/bus/motorcycle» obýektiniň
  görnüşine öwredilen, reňk oňa möhüm däl. Reňki biz goşmaça özümiz
  *kesgitleýäris* (HSV-ewristika) we hasabata ýazýarys.
* **Sütünler awtomatiki** — «fon» kadrda (wideo boýunça mediana, awtoulaglar
  ýitýär) uzyn dik gyralary gözleýäris (Canny + Hough), X boýunça toparlaýarys.
  Tapylan ýerleşişi interfeýsde düzedip bolýar.
* **Wagt boýunça takyklyk** — çyzykdan geçen pursat iki kadryň arasynda
  interpolýasiýa bilen hasaplanýar, kadra çenli tegeleklenmeýär.
* **Neýron torsuz** ätiýaçlyk režim (MOG2 fon aýyrmak) — testler we internetsiz
  kompýuterler üçin.

## Gurnamak

```bash
cd video-detect-program
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ilkinji işletmede YOLO modeli `yolov8n.pt` (~6 MB) awtomatiki ýüklenýär.
Internet çäkli bolsa — faýly öňünden ýükläp, `app.py`-yň ýanynda goýuň:
<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt>

## Işe girizmek

### Web-interfeýs (esasy usul)

```bash
streamlit run app.py
```

1. Wideo ýükläň.
2. Tapylan sütünleri barlaň (sary çyzyklar P1…Pn); gerek bolsa duýgurlygy
   üýtgediň ýa-da X sanawyny düzediň.
3. Sütünleriň arasyndaky aralyklary giriziň (bir san «hemmesi üçin» ýa-da her
   biri aýratyn).
4. «Tizligi hasapla» → tablisa, grafik, bellikli wideo, CSV / JSON.
5. «Kalibrowkany ýükle» düwmesi `calibration.json` (sütünler + aralyklar)
   saklaýar — soňra kamera režimi ony ulanýar.

### Konsol / kamera

```bash
# wideo faýl, sütünler awtomatiki, hemmesiniň arasy 10 m
python cli.py --source road.mp4 --distances 10 --show

# dürli aralyklar (4 sütün → 3 aralyk), CSV we wideo sakla
python cli.py --source road.mp4 --distances 12,15.5,12 --csv out.csv --video out.mp4

# web-interfeýsdäki kalibrowka bilen janly USB-kamera
python cli.py --source 0 --config calibration.json --show --limit 60

# IP-kamera (RTSP)
python cli.py --source "rtsp://user:pass@192.168.1.10:554/stream" --config calibration.json --show
```

Penjirede `q` ýa-da `Esc` — togtatmak.

### Hakyky wideosyz test

```bash
python samples/make_synthetic.py            # samples/synthetic.mp4 döredýär
python cli.py --source samples/synthetic.mp4 --distances 10 --detector motion
```

Böleklerdäki hakyky tizlikler 36 / 54 / 72 km/sag; programma ≈35–36 / 54 / 72 berýär.

**Möhüm:** `synthetic.mp4`-de «awtoulag» — çyzylan gönüburçluk. YOLO hakyky
awtoulaglaryň suratlaryna öwredilen we beýle çyzgyny awtoulag hasaplamaýar,
şonuň üçin bu faýlda YOLO «hiç bir awtoulag iki sütünden geçmedi» diýer. Hut
YOLO-ny barlamak üçin ikinji wideo bar — onda ýoldan **hakyky ulagyň suraty**
geçýär:

```bash
python samples/make_synthetic.py --sprite samples/real_vehicle.jpg --out samples/realistic.mp4
python cli.py --source samples/realistic.mp4 --distances 10 --detector yolo
```

YOLO-nyň ondaky netijesi: 34.5 / 54.3 / 66 km/sag (soňky bölek pes — obýekt giň
we kadryň gyrasyna direýär; hakyky düşürilişde awtoulag kadryň kiçi bölegini
tutýar we bu mesele bolmaýar). Hakyky wideoda YOLO ulanyň.

## Esasy.exe — Windows üçin gurnamasyz wersiýa

`Esasy.exe` — ähli kitaphanalar (Python, torch, YOLO, OpenCV, Streamlit) we
`yolov8n.pt` modeli içinde. Python gurnamak gerek däl. Iki gezek basylanda
programma aýratyn penjirede açylýar (Windows-yň öz Edge WebView2 komponenti
ulanylýar; ol Windows 10/11-de öňünden bar). Ölçegi ~1 GB (zip ~400–500 MB).

Windows-exe diňe Windows-da ýygnalýar, şonuň üçin iki ýol bar:

**1. GitHub Actions (awtomatiki).** Proýekti GitHub-a goýuň — her `push`-dan
soň `.github/workflows/build-windows.yml` Windows-serwerde exe-ni ýygnaýar,
özüni barlaýar (`--selftest`, YOLO-ly doly geçiş, serweriň açylyşy) we
`Esasy-windows.zip` artefaktyny goýýar (Actions → işleýiş → Artifacts).
`v1.0` ýaly teg goýsaňyz — Releases sahypasynda hemişelik zip döreýär.

```bash
cd video-detect-program
git init && git add . && git commit -m "SpeedCam"
git branch -M main
git remote add origin https://github.com/<ulanyjy>/<repo>.git
git push -u origin main            # → Actions-da ýygnalyş başlaýar (~20–30 min)
git tag v1.0 && git push origin v1.0   # → Releases-de Esasy-windows.zip
```

Repozitoriýany **public** ediň ýa-da Releases ulanyň: private repoda
artefaktlaryň mugt ýeri 500 MB bilen çäkli.

**2. Islendik Windows kompýuterde el bilen:** Python 3.10–3.12 gurnalan bolsa,
`build_windows.bat` işlediň → `dist\Esasy\Esasy.exe`. Tutuş `dist\Esasy`
papkasyny zip edip paýlaň (exe ýeke özi işlemeýär, `_internal` gerek).

Exe-niň goşmaça režimleri (buýruk setirinden):

```bat
Esasy.exe --selftest                                   :: kitaphanalar we model barmy
Esasy.exe --cli --source road.mp4 --distances 10       :: konsol režimi (cli.py ýaly)
Esasy.exe --cli --source 0 --config calibration.json --show   :: janly kamera
```

Žurnal: `%LOCALAPPDATA%\SpeedCam\server.log` (penjire açylmasa — şoňa serediň).

## Gurluşy

```
app.py                  web-interfeýs (Streamlit)
cli.py                  konsol / kamera
Esasy.py                Esasy.exe-niň başlangyjy (serwer + aýratyn penjire)
Esasy.spec              PyInstaller konfigurasiýasy
build_windows.bat       Windows-da el bilen ýygnamak
.github/workflows/      GitHub Actions: Windows-exe awtomatiki ýygnalyş
speedcam/
  video.py              kadr çeşmesi (faýl / kamera / RTSP), fon
  detector.py           YoloDetector (ultralytics + ByteTrack), MotionDetector (MOG2)
  poles.py              sütünleri awtomatiki gözlemek
  speed.py              çyzyklardan geçmek we tizligi hasaplamak
  color.py              awtoulagyň reňkini kesgitlemek, türkmençe atlar
  paths.py              faýl ýollary (adaty / exe içinde)
  pipeline.py           konweýer, çyzgy, CSV/JSON eksport
samples/make_synthetic.py   test wideo generatory
samples/real_vehicle.jpg    YOLO testi üçin hakyky ulagyň suraty
```

## Takyk işlemegi üçin nähili düşürmeli

* Kamera hereketsiz (ştatiw), ýola **perpendikulýar** seredýär.
* Ähli sütünler kadrda, awtoulag **sütünler bilen deň uzaklykda** geçýär —
  ýogsam perspektiwa aralyklary ýoýýar.
* Sütünleriň arasyndaky aralyk rulletka / resminama boýunça ölçelen.
* FPS näçe ýokary bolsa, şonça takyk: 30 FPS we 10 m-de 60 km/sag-da ýalňyşlyk
  ≈ ±1–2 km/sag.

## Prototipiň çäklendirmeleri we indiki ädimler

* Perspektiwa düzedilmeýär (gapdaldan düşürmek gerek). Burç astyndaky kamera
  üçin — gomografiýa goşmaly (ýolda 4 nokat → metrler).
* Sütünleriň bir «hatary». Awtoulaglar dürli zolaklarda iki tarapa gidýän
  bolsa — her zolak üçin çyzyklary aýratyn bermeli.
* Gowşak kompýuterde janly kamera üçin — `yolov8n`, `imgsz`-i kiçeltmek, her
  2-nji kadry işlemek.
* Indiki ädimler: belgini tanamak (EasyOCR), maglumat bazasyna ýazmak,
  çäkden geçmek barada Telegram-habarnama, günler boýunça statistika paneli.
