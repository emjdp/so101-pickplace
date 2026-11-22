"""탑뷰 카메라 픽셀 (u, v) ↔ 로봇 베이스 평면 좌표 (x, y) 변환.

계획서 3.2절 시스템 A의 카메라-로봇 좌표 보정:
1. 호모그래피 보정 (`HomographyCalib`):
   - 작업대 위 기준점 4~9개에 그리퍼 끝을 대고, FK로 구한 로봇 베이스 좌표 (x, y)와
     영상에서 찍은 픽셀 좌표 (u, v)를 짝지어 3×3 평면 호모그래피 행렬 H를 구한다.
   - 필요 시 카메라 내부 파라미터(K, dist)로 렌즈 왜곡을 먼저 편 뒤 호모그래피를 적용한다.
   - 블록 높이가 기준점 측정 높이(테이블면 z0)보다 h 만큼 높을 때, 카메라 광축 기울기에 의한
     시차(parallax)를 보정하거나 블록 윗면 높이에서 직접 호모그래피를 구할 수 있다.

2. 3D 광선-평면 교차 (`TopViewRayCalib`):
   - 카메라 내부 파라미터(K, dist)와 베이스 기준 카메라 외부 자세(C, R)를 알 때,
     픽셀 (u, v)에서 나가는 3D 광선이 높이 z 평면과 만나는 점 P = C + (z - Cz) / rz * r 을 계산한다.
   - `to_homography(z)`로 높이 z에서의 호모그래피 행렬 H를 구할 수 있다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from so101.calib import load_intrinsics


def _undistort_uv(uv: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    """픽셀 좌표 배열 (N, 2)를 K, dist 로 왜곡 보정해 다시 픽셀 좌표 (N, 2)로 돌려준다."""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if K is None or dist is None:
        return pts.reshape(-1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def fit_homography(
    uv_px: np.ndarray,
    xy_m: np.ndarray,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    ransac_thresh_m: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """픽셀 좌표 (N, 2) → 로봇 베이스 평면 좌표 (N, 2) [m] 호모그래피 행렬 H (3, 3)를 계산한다.

    반환: (H, errors_m) — errors_m 은 각 기준점의 변환 오차 [m] 배열 (N,).
    """
    uv = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
    if len(uv) < 4 or len(uv) != len(xy):
        raise ValueError(f"호모그래피는 최소 4쌍의 대응점이 필요하다 (입력: {len(uv)}개)")

    uv_u = _undistort_uv(uv, K, dist)
    method = cv2.RANSAC if len(uv) >= 5 else 0
    H, _ = cv2.findHomography(uv_u, xy, method=method, ransacReprojThreshold=ransac_thresh_m)
    if H is None:
        raise RuntimeError("호모그래피 행렬 H를 구하지 못함 (점들이 한 직선 위에 있는지 확인)")

    pred = apply_homography(H, uv, K=K, dist=dist)
    errors = np.linalg.norm(pred - xy, axis=1)
    return H, errors


def apply_homography(
    H: np.ndarray,
    uv_px,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """호모그래피 H로 픽셀 (u, v) 또는 (N, 2) 배열을 로봇 베이스 (x, y) [m]로 변환한다."""
    arr = np.asarray(uv_px, dtype=np.float64)
    single = arr.ndim == 1
    uv_u = _undistort_uv(arr, K, dist)
    ones = np.ones((len(uv_u), 1), dtype=np.float64)
    hom = (np.asarray(H, dtype=np.float64) @ np.hstack([uv_u, ones]).T).T
    xy = hom[:, :2] / hom[:, 2:3]
    return xy[0] if single else xy


def apply_homography_inverse(
    H: np.ndarray,
    xy_m,
) -> np.ndarray:
    """로봇 베이스 (x, y) [m] → 왜곡 보정된 픽셀 (u, v) 역변환."""
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    arr = np.asarray(xy_m, dtype=np.float64)
    single = arr.ndim == 1
    pts = arr.reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = (H_inv @ np.hstack([pts, ones]).T).T
    uv = hom[:, :2] / hom[:, 2:3]
    return uv[0] if single else uv


def pixel_angle_to_world_rad(
    H: np.ndarray,
    u: float,
    v: float,
    angle_deg: float,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> float:
    """영상에서 검출한 블록 회전각(deg)을 호모그래피 H를 통해 로봇 베이스 평면의 각도(rad)로 변환한다."""
    rad = math.radians(angle_deg)
    du, dv = 20.0 * math.cos(rad), 20.0 * math.sin(rad)
    p0 = apply_homography(H, [u, v], K=K, dist=dist)
    p1 = apply_homography(H, [u + du, v + dv], K=K, dist=dist)
    d = p1 - p0
    return math.atan2(float(d[1]), float(d[0]))


@dataclass
class HomographyCalib:
    """호모그래피 기반 카메라-로봇 좌표 변환기."""

    H: np.ndarray
    plane_z_m: float = 0.0
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    rms_mm: float = 0.0
    max_mm: float = 0.0

    def pixel_to_xy(self, u: float, v: float) -> tuple[float, float]:
        xy = apply_homography(self.H, [u, v], K=self.K, dist=self.dist)
        return float(xy[0]), float(xy[1])

    def xy_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        uv = apply_homography_inverse(self.H, [x, y])
        return float(uv[0]), float(uv[1])

    def pixel_angle_to_world(self, u: float, v: float, angle_deg: float) -> float:
        return pixel_angle_to_world_rad(self.H, u, v, angle_deg, K=self.K, dist=self.dist)

    def save(self, path: Path, points: list[dict] | None = None, intrinsics_path: Path | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "plane_z_m": float(self.plane_z_m),
            "undistorted": self.K is not None,
            "intrinsics": str(intrinsics_path) if intrinsics_path else None,
            "rms_mm": round(float(self.rms_mm), 2),
            "max_mm": round(float(self.max_mm), 2),
            "H": np.asarray(self.H, dtype=float).tolist(),
            "points": points or [],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path, intrinsics_path: Path | None = None) -> HomographyCalib:
        data = json.loads(Path(path).read_text())
        K = dist = None
        if data.get("undistorted"):
            intr = intrinsics_path or (Path(data["intrinsics"]) if data.get("intrinsics") else None)
            if intr is not None and Path(intr).exists():
                K, dist = load_intrinsics(intr)
        return cls(
            H=np.array(data["H"], dtype=np.float64),
            plane_z_m=float(data.get("plane_z_m", 0.0)),
            K=K,
            dist=dist,
            rms_mm=float(data.get("rms_mm", 0.0)),
            max_mm=float(data.get("max_mm", 0.0)),
        )


class TopViewRayCalib:
    """카메라 외부 자세(C, R)와 내부 파라미터(K, dist)를 이용한 3D 광선-평면 교차 변환기.

    `to_homography(z)`는 높이 z에서의 호모그래피를 계산한다.
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray, xyz_m, rotation):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.C = np.asarray(xyz_m, dtype=np.float64)
        self.R = np.asarray(rotation, dtype=np.float64)

    @classmethod
    def from_files(cls, intrinsics_path: Path, extrinsics_path: Path) -> TopViewRayCalib:
        K, dist = load_intrinsics(intrinsics_path)
        ext = json.loads(Path(extrinsics_path).read_text())
        return cls(K, dist, ext["xyz_m"], ext["rotation"])

    def ray(self, u: float, v: float) -> np.ndarray:
        n = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.dist).ravel()
        r = self.R @ np.array([n[0], n[1], 1.0], dtype=np.float64)
        return r / np.linalg.norm(r)

    def to_plane(self, u: float, v: float, z: float) -> np.ndarray:
        r = self.ray(u, v)
        if abs(r[2]) < 1e-6:
            raise ValueError("광선이 작업 평면과 평행함")
        return self.C + (z - self.C[2]) / r[2] * r

    def project(self, p) -> tuple[float, float]:
        pc = self.R.T @ (np.asarray(p, dtype=np.float64) - self.C)
        uv, _ = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return float(uv[0, 0, 0]), float(uv[0, 0, 1])

    def to_homography(self, plane_z_m: float = 0.0) -> HomographyCalib:
        """높이 plane_z_m 인 수평면과 카메라 영상 사이의 정확한 평면 호모그래피 H를 유도한다."""
        grid_xy = np.array(
            [[0.18, -0.12], [0.18, 0.12], [0.32, 0.12], [0.32, -0.12], [0.25, 0.0]],
            dtype=np.float64,
        )
        uv = np.array([self.project([x, y, plane_z_m]) for x, y in grid_xy], dtype=np.float64)
        H, errs = fit_homography(uv, grid_xy, K=self.K, dist=self.dist)
        return HomographyCalib(
            H=H,
            plane_z_m=plane_z_m,
            K=self.K,
            dist=self.dist,
            rms_mm=float(np.sqrt(np.mean(errs**2)) * 1000),
            max_mm=float(np.max(errs) * 1000),
        )
