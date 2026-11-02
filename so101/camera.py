"""카메라 열기 도우미. (실습 대상 아님)"""

import cv2

import config


def open_camera(dev=config.TOP_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fourcc=config.CAM_FOURCC):
    cap = cv2.VideoCapture(str(dev), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없음: {dev}  (ls -l /dev/soarm_* 로 링크 확인)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 처음 몇 프레임은 밝기가 안정되지 않아서 버린다
    for _ in range(5):
        cap.read()
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("프레임을 읽지 못함")
    return frame


def require_gui():
    """cv2.imshow 가 되는 OpenCV 인지 확인한다. headless 빌드면 안내하고 종료."""
    if "GUI:                           NONE" in cv2.getBuildInformation():
        raise SystemExit(
            "이 OpenCV는 headless 빌드라 창을 띄울 수 없다.\n"
            "  pip uninstall -y opencv-python-headless && pip install opencv-python==4.13.0.92\n"
            "(문서 '1. 환경과 LeRobot' 0단계 참고)"
        )


def draw_grid(img, n=4, color=(255, 255, 0)):
    h, w = img.shape[:2]
    for i in range(1, n):
        cv2.line(img, (w * i // n, 0), (w * i // n, h), color, 1)
        cv2.line(img, (0, h * i // n), (w, h * i // n), color, 1)
    return img
