"""색 블록 검출. ★ 실습: TODO 를 채운다 ★

확인: pytest tests/test_detect.py -v
      python scripts/detect_live.py
참고: 문서 '3. 카메라 보정과 블록 검출' 3절

흐름
  BGR 영상 → HSV 변환 → 색 범위 마스크 → 잡음 제거 → 가장 큰 윤곽선 → 중심 (u, v), 회전각
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    u: float  # 중심 픽셀 x (열)
    v: float  # 중심 픽셀 y (행)
    angle_deg: float  # 블록 회전각. 정사각형이면 90° 마다 같은 모양이라 [0, 90) 로 맞춘다
    area: float  # 윤곽선 넓이 [px^2]
    box: np.ndarray  # minAreaRect 꼭짓점 4개 (4, 2), 그리기용


def make_mask(bgr: np.ndarray, hsv_ranges: list[tuple]) -> np.ndarray:
    """색 범위 안의 픽셀은 255, 나머지는 0 인 마스크 (H, W) uint8.

    hsv_ranges: [(lo, hi), ...]  lo, hi 는 (H, S, V). OpenCV 에서 H 는 0~179 이다.
                범위가 여러 개면 합친다 (빨간색은 H 가 0 근처와 179 근처로 나뉘어서 2개가 필요).

    힌트: cv2.cvtColor(..., cv2.COLOR_BGR2HSV), cv2.inRange, cv2.bitwise_or
          마지막에 cv2.morphologyEx 로 OPEN(점 잡음 제거) → CLOSE(구멍 메우기)
    """
    raise NotImplementedError("TODO: make_mask")


def find_block(mask: np.ndarray, min_area: float = 150.0) -> Detection | None:
    """마스크에서 가장 큰 덩어리를 블록으로 보고 Detection 을 만든다. 없으면 None.

    힌트: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          cv2.contourArea 로 가장 큰 것 고르기, min_area 보다 작으면 None
          중심: cv2.moments 의 m10/m00, m01/m00  (minAreaRect 의 중심과 비교해 볼 것)
          각도: cv2.minAreaRect → ((cx, cy), (w, h), angle). angle % 90 으로 [0, 90) 에 맞춘다
          box : cv2.boxPoints(rect)
    """
    raise NotImplementedError("TODO: find_block")


def detect_block(bgr: np.ndarray, hsv_ranges: list[tuple], min_area: float = 150.0) -> Detection | None:
    return find_block(make_mask(bgr, hsv_ranges), min_area)


def draw_detection(img: np.ndarray, det: Detection | None) -> np.ndarray:
    if det is None:
        cv2.putText(img, "no block", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    cv2.drawContours(img, [det.box.astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.circle(img, (int(det.u), int(det.v)), 4, (0, 0, 255), -1)
    text = f"u={det.u:.1f} v={det.v:.1f} ang={det.angle_deg:.1f} area={det.area:.0f}"
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
