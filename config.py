"""장치 경로, ID, 파일 위치를 한곳에 모아 둔다. 장치가 바뀌면 여기만 고친다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 로봇 팔 (udev/99-soarm.rules 로 만든 고정 링크) ─────────────────
FOLLOWER_PORT = "/dev/soarm_follower"
LEADER_PORT = "/dev/soarm_leader"

# 캘리브레이션 파일 이름이 된다: calib/lerobot/follower/tp_follower.json
FOLLOWER_ID = "tp_follower"
LEADER_ID = "tp_leader"
FOLLOWER_CALIB_DIR = ROOT / "calib" / "lerobot" / "follower"
LEADER_CALIB_DIR = ROOT / "calib" / "lerobot" / "leader"

# ── 탑뷰 카메라 (Logitech C270) ─────────────────────────────────────
TOP_CAM = "/dev/soarm_cam_top"
CAM_WIDTH = 640
CAM_HEIGHT = 480
# C270의 MJPG는 "Corrupt JPEG data" 경고가 자주 뜬다. 카메라 1대면 YUYV로 충분하다.
CAM_FOURCC = "YUYV"

CAMERA_INTRINSICS = ROOT / "calib" / "camera_top.yaml"

# ── 로봇 모델 ───────────────────────────────────────────────────────
# git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
URDF_PATH = ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"

DATA_DIR = ROOT / "data"
