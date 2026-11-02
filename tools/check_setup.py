"""세션 시작 전 환경·장치 점검. 하나씩 ✅/⚠️/❌ 로 보여 준다.

python tools/check_setup.py
"""

import glob
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

results = []


def check(name, ok, detail="", hint="", warn_only=False):
    mark = "✅" if ok else ("⚠️ " if warn_only else "❌")
    results.append(ok or warn_only)
    print(f"{mark} {name:28s} {detail}")
    if not ok and hint:
        print(f"   → {hint}")


# ── 파이썬 환경 ──
check("Python 3.12+", sys.version_info >= (3, 12), sys.version.split()[0], "conda activate lerobot")
check("ROS 2 꺼짐", not os.environ.get("ROS_DISTRO"), os.environ.get("ROS_DISTRO", ""), "ROS 2 Switch 로 끄고 새 터미널")

try:
    lerobot = importlib.import_module("lerobot")
    check("lerobot", True, getattr(lerobot, "__version__", "?"))
except ImportError as e:
    check("lerobot", False, str(e), "conda activate lerobot")

try:
    import cv2

    gui = "GUI:                           NONE" not in cv2.getBuildInformation()
    check("OpenCV (창 띄우기)", gui, cv2.__version__, "pip uninstall -y opencv-python-headless && pip install opencv-python==4.13.0.92")
except ImportError as e:
    check("OpenCV", False, str(e))

for mod, why in [("pytest", "빈칸 채점"), ("yourdfpy", "FK 교차 검증"), ("matplotlib", "FK 막대 그림")]:
    try:
        importlib.import_module(mod)
        check(mod, True, why)
    except ImportError:
        check(mod, False, why, "pip install pytest yourdfpy matplotlib", warn_only=True)

# ── 파일 ──
check("URDF", config.URDF_PATH.exists(), str(config.URDF_PATH.relative_to(config.ROOT)),
      "git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100")
for name, d, i in [("팔로워 캘리브레이션", config.FOLLOWER_CALIB_DIR, config.FOLLOWER_ID),
                   ("리더 캘리브레이션", config.LEADER_CALIB_DIR, config.LEADER_ID)]:
    f = d / f"{i}.json"
    check(name, f.exists(), str(f.relative_to(config.ROOT)), "문서 '1. 환경과 LeRobot' 3단계 lerobot-calibrate", warn_only=True)

# ── 장치 ──
acm = sorted(glob.glob("/dev/ttyACM*"))
for name, port in [("팔로워 포트", config.FOLLOWER_PORT), ("리더 포트", config.LEADER_PORT)]:
    exists = os.path.exists(port)
    detail = f"{port} -> {os.path.realpath(port)}" if exists else f"{port} 없음 (보이는 포트: {acm or '없음'})"
    check(name, exists, detail, "USB·전원 확인 → udev 규칙 적용: sudo cp udev/99-soarm.rules /etc/udev/rules.d/ ...")

if os.path.exists(config.TOP_CAM):
    try:
        from so101.camera import open_camera, read_frame

        cap = open_camera()
        frame = read_frame(cap)
        cap.release()
        check("탑뷰 카메라", True, f"{config.TOP_CAM} {frame.shape[1]}x{frame.shape[0]} 평균밝기 {frame.mean():.0f}")
    except Exception as e:  # noqa: BLE001
        check("탑뷰 카메라", False, str(e), "다른 프로그램이 카메라를 쓰고 있는지 확인")
else:
    check("탑뷰 카메라", False, f"{config.TOP_CAM} 없음", "C270 연결 → udev 규칙 적용 (v4l2-ctl --list-devices)")

print()
print("모두 통과" if all(results) else "❌ 항목부터 해결하고 다시 실행")
