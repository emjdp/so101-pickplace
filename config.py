"""장치 경로, ID, 파일 위치를 한곳에 모아 둔다. 장치가 바뀌면 여기만 고친다."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── 로봇 팔 (udev/99-soarm.rules 로 만든 고정 링크) ─────────────────
def _arm_port(link, serial):
    """udev 링크가 없는 컴퓨터(규칙을 안 깐 rpi5 등)에서는 같은 보드의 by-id 경로를 쓴다."""
    by_id = f"/dev/serial/by-id/usb-1a86_USB_Single_Serial_{serial}-if00"
    return by_id if not os.path.exists(link) and os.path.exists(by_id) else link


FOLLOWER_PORT = _arm_port("/dev/soarm_follower", "5B7B015254")
LEADER_PORT = _arm_port("/dev/soarm_leader", "5B7B014404")

# 캘리브레이션 파일 이름이 된다: calib/lerobot/follower/tp_follower.json
FOLLOWER_ID = "tp_follower"
LEADER_ID = "tp_leader"
FOLLOWER_CALIB_DIR = ROOT / "calib" / "lerobot" / "follower"
LEADER_CALIB_DIR = ROOT / "calib" / "lerobot" / "leader"

# 원격 텔레옵 (scripts/teleop_remote.py): 리더는 노트북, 팔로워는 Raspberry Pi 5
FOLLOWER_HOST = "rpi5.local"
TELEOP_UDP_PORT = 5560

# ── 탑뷰 카메라 (Logitech C270) 및 손목 카메라 (icSpring) ────────────
TOP_CAM = "/dev/soarm_cam_top"
WRIST_CAM = "/dev/soarm_cam_wrist"
CAM_WIDTH = 640
CAM_HEIGHT = 480
# C270의 MJPG는 "Corrupt JPEG data" 경고가 자주 뜬다. 카메라 1대면 YUYV로 충분하다.
# (ACT 녹화·추론처럼 카메라 2대를 동시에 쓸 때는 USB 대역폭 때문에 MJPG를 쓴다)
CAM_FOURCC = "YUYV"

CAMERA_INTRINSICS = ROOT / "calib" / "camera_top.yaml"
CAMERA_EXTRINSICS = ROOT / "calib" / "camera_extrinsics.json"
HOMOGRAPHY_CALIB = ROOT / "calib" / "homography.json"

# ── 작업 영역 및 상자 기본 위치 (base_link 기준 [m]) ────────────────
TABLE_Z_M = 0.0
BLOCK_HEIGHT_M = 0.025
BOX_XYZ_M = (0.198, -0.074, 0.065)
WORKSPACE_X_M = (0.12, 0.33)
WORKSPACE_Y_M = (-0.12, 0.14)

# ── 시스템 B (ACT 모방학습) 기본 설정 ───────────────────────────────
ACT_REPO_ID = "emjdp/so101_pickplace"
ACT_DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / ACT_REPO_ID
ACT_TASK = "Pick up the block and put it in the box"
ACT_CHECKPOINT_DIR = ROOT / "outputs" / "act_pickplace" / "pretrained_model"

# ── 로봇 모델 ───────────────────────────────────────────────────────
# git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
URDF_PATH = ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"

DATA_DIR = ROOT / "data"
EXPERIMENT_CSV = DATA_DIR / "experiments" / "trials.csv"
