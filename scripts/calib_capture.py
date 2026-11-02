"""체커보드 사진을 모은다 (카메라 내부 보정용).

python scripts/calib_capture.py                   # 내부 코너 9x6
python scripts/calib_capture.py --cols 8 --rows 5

  Space: 코너가 검출된 상태에서 저장 (data/calib/)   q: 종료

잘 모으는 법 (문서 '3. 카메라 보정과 블록 검출' 4절)
- 15~25장. 화면 가운데만이 아니라 네 귀퉁이·가장자리까지 골고루 (왜곡은 가장자리에서 크다)
- 보드를 앞뒤·좌우로 20~40° 기울인 사진을 섞는다. 전부 정면이면 초점거리가 잘 안 잡힌다
- 흔들린 사진은 버린다. 노출은 lock_camera.sh 로 고정해 둔다
화면의 노란 점은 지금까지 저장한 사진들의 코너 위치다. 빈 곳을 채우듯 찍는다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402

import config  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
args = ap.parse_args()
require_gui()

out_dir = config.DATA_DIR / "calib"
out_dir.mkdir(parents=True, exist_ok=True)
pattern = (args.cols, args.rows)
saved_corners = []
n_saved = len(list(out_dir.glob("*.jpg")))
cap = open_camera()
while True:
    frame = read_frame(cap)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)
    view = frame.copy()
    for c in saved_corners:
        for p in c[::4]:
            cv2.circle(view, (int(p[0][0]), int(p[0][1])), 2, (0, 255, 255), -1)
    if found:
        cv2.drawChessboardCorners(view, pattern, corners, found)
    cv2.putText(view, f"saved {n_saved}  {'FOUND' if found else 'not found'}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("calib capture (space: save, q: quit)", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and found:
        path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{n_saved:02d}.jpg"
        cv2.imwrite(str(path), frame)  # 그림이 덧그려지지 않은 원본을 저장
        saved_corners.append(corners)
        n_saved += 1
        print("저장:", path)
cap.release()
cv2.destroyAllWindows()
