"""인쇄용 체커보드 PDF 를 만든다 (A4 가로, 300 dpi).

python tools/make_checkerboard.py                 # 내부 코너 9x6, 한 칸 25 mm
python tools/make_checkerboard.py --cols 8 --rows 5 --square-mm 30

인쇄할 때 '실제 크기(100%)' 로 뽑고, 인쇄된 칸 크기를 자로 다시 잰다.
보정에는 잰 값을 쓴다 (프린터가 몇 % 줄여서 뽑는 경우가 많다).
판에 붙여 평평하게 만들 것. 휘어 있으면 보정 결과가 나빠진다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cols", type=int, default=9, help="가로 내부 코너 수")
ap.add_argument("--rows", type=int, default=6, help="세로 내부 코너 수")
ap.add_argument("--square-mm", type=float, default=25.0)
args = ap.parse_args()

DPI = 300
px = lambda mm: round(mm / 25.4 * DPI)  # noqa: E731
W, H = px(297), px(210)
nx, ny = args.cols + 1, args.rows + 1  # 칸 수 = 내부 코너 + 1
s = px(args.square_mm)
if nx * s > W - px(20) or ny * s > H - px(20):
    raise SystemExit("A4 에 안 들어간다. --square-mm 를 줄일 것")

img = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(img)
x0, y0 = (W - nx * s) // 2, (H - ny * s) // 2
for j in range(ny):
    for i in range(nx):
        if (i + j) % 2 == 0:
            d.rectangle([x0 + i * s, y0 + j * s, x0 + (i + 1) * s - 1, y0 + (j + 1) * s - 1], fill=0)
d.text((x0, y0 + ny * s + px(4)), f"inner corners {args.cols}x{args.rows}, square {args.square_mm} mm @ 100% scale", fill=0)

out = config.DATA_DIR / f"checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
Image.init()  # PDF 저장에 필요한 플러그인을 불러온다
img.save(out, resolution=DPI)
print("저장:", out)
