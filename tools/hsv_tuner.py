"""블록 색의 HSV 범위를 찾는다.

python tools/hsv_tuner.py blue        # 결과는 calib/hsv_blue.json
python tools/hsv_tuner.py red --ranges 2   # 빨강은 H 가 0 근처와 179 근처로 나뉜다

- 왼쪽 창(카메라)에서 블록을 클릭하면 그 점의 HSV 가 터미널에 찍힌다. 범위를 잡는 출발점으로 쓴다.
- 트랙바로 범위를 조절하면서 오른쪽 마스크에 블록만 하얗게 남게 만든다.
- 블록을 작업 영역 여러 곳(가장자리, 그림자 쪽)에 옮겨 가며 확인한다.
  s: 저장   n: (--ranges 2 일 때) 편집할 범위 바꾸기   q: 종료
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("color")
ap.add_argument("--ranges", type=int, default=1, choices=[1, 2])
args = ap.parse_args()
require_gui()

out = config.ROOT / "calib" / f"hsv_{args.color}.json"
# 저장된 값이 있으면 이어서 조정한다
ranges = json.loads(out.read_text())["ranges"][: args.ranges] if out.exists() else []
while len(ranges) < args.ranges:
    ranges.append([[0, 100, 60], [179, 255, 255]])
cur = 0

WIN = "mask"
cv2.namedWindow(WIN)
names = [("H lo", 179), ("H hi", 179), ("S lo", 255), ("S hi", 255), ("V lo", 255), ("V hi", 255)]


def load_trackbars():
    lo, hi = ranges[cur]
    vals = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
    for (n, _), v in zip(names, vals):
        cv2.setTrackbarPos(n, WIN, int(v))


for n, mx in names:
    cv2.createTrackbar(n, WIN, 0, mx, lambda _: None)
load_trackbars()

cap = open_camera()
hsv = None


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and hsv is not None:
        print(f"(u={x}, v={y}) HSV = {hsv[y, x].tolist()}")


cv2.namedWindow("camera")
cv2.setMouseCallback("camera", on_click)

while True:
    frame = read_frame(cap)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = [cv2.getTrackbarPos(n, WIN) for n, _ in names]
    ranges[cur] = [[v[0], v[2], v[4]], [v[1], v[3], v[5]]]
    mask = np.zeros(frame.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    cv2.putText(frame, f"editing range {cur + 1}/{len(ranges)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("camera", frame)
    cv2.imshow(WIN, mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("n") and len(ranges) > 1:
        cur = (cur + 1) % len(ranges)
        load_trackbars()
    if key == ord("s"):
        out.write_text(json.dumps({"color": args.color, "ranges": ranges}, indent=2))
        print("저장:", out, ranges)
cap.release()
cv2.destroyAllWindows()
