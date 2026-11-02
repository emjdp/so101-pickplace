"""탑뷰 카메라 프리뷰. 노출을 맞추거나 화면 구도를 잡을 때 쓴다.

python tools/camera_preview.py
  s: 스냅샷 저장 (data/snapshots/)   g: 격자 켜기/끄기   q: 종료

화면 왼쪽 위 숫자
  mean: 평균 밝기 (0~255). 100~160 정도면 무난하다
  sat:  255 근처로 하얗게 날아간 픽셀 비율. 1% 넘으면 노출을 줄인다
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402

import config  # noqa: E402
from so101.camera import draw_grid, open_camera, read_frame, require_gui  # noqa: E402

require_gui()
out_dir = config.DATA_DIR / "snapshots"
out_dir.mkdir(parents=True, exist_ok=True)
cap = open_camera()
grid = True
while True:
    frame = read_frame(cap)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    view = frame.copy()
    if grid:
        draw_grid(view)
    sat = (gray > 250).mean() * 100
    cv2.putText(view, f"mean {gray.mean():.0f}  sat {sat:.1f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("top camera (s: save, g: grid, q: quit)", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("g"):
        grid = not grid
    if key == ord("s"):
        path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        print("저장:", path)
cap.release()
cv2.destroyAllWindows()
