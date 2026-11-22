"""시스템 B (ACT 모방학습) 정책 로더 및 추론 실행기.

계획서 3.3절 시스템 B:
  리더암 텔레오퍼레이션 -> 데모 기록(탑뷰/손목 영상, 관절각) -> ACT 학습(Seraph) -> 팔로워암 정책 실행

LeRobot ACT 체크포인트를 불러오고, 지정한 장치에서 추론하도록 전처리기의 장치 설정을 바꾼다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from so101.robot import ALL_MOTORS, joint_positions


@dataclass
class ACTRolloutResult:
    """ACT 정책 1회 에피소드 실행 결과."""

    steps: int
    elapsed_s: float
    final_joints: dict[str, float]


class ACTRunner:
    """학습된 LeRobot ACT 체크포인트를 불러와 관측(관절각 + 카메라 영상)에서 행동(목표 관절각)을 추론한다."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu"):
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if (self.checkpoint_dir / "pretrained_model").is_dir():
            self.checkpoint_dir = self.checkpoint_dir / "pretrained_model"
        if not (self.checkpoint_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"ACT 체크포인트를 찾을 수 없음: {self.checkpoint_dir}")

        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(self.checkpoint_dir)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.checkpoint_dir,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.camera_keys = [
            k.removeprefix("observation.images.")
            for k in self.policy.config.input_features
            if k.startswith("observation.images.")
        ]

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @staticmethod
    def _image_to_tensor(bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        img = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB) if is_bgr else bgr_or_rgb
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    def predict(
        self,
        joints_deg: dict[str, float],
        images_bgr: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """현재 관절값 {관절명: deg} 과 카메라 BGR 프레임 {카메라명: (H, W, 3)} 으로 다음 목표 관절값을 반환한다."""
        state = torch.tensor(
            [[float(joints_deg[m]) for m in ALL_MOTORS]],
            dtype=torch.float32,
        )
        batch: dict[str, torch.Tensor] = {"observation.state": state}
        for cam in self.camera_keys:
            if cam not in images_bgr:
                raise KeyError(f"정책이 요구하는 카메라 '{cam}' 영상이 입력에 없음 (요구: {self.camera_keys})")
            batch[f"observation.images.{cam}"] = self._image_to_tensor(images_bgr[cam], is_bgr=True)

        with torch.inference_mode():
            proc = self.preprocessor(batch)
            action = self.policy.select_action(proc)
            action = self.postprocessor(action)
        vals = action.squeeze(0).cpu().numpy().tolist()
        return {m: float(v) for m, v in zip(ALL_MOTORS, vals)}


def run_act_episode(
    runner: ACTRunner,
    robot,
    cameras: dict[str, object],
    max_duration_s: float = 30.0,
    fps: float = 30.0,
    on_step=None,
) -> ACTRolloutResult:
    """로봇과 카메라 딕셔너리({이름: cv2.VideoCapture 또는 프레임 함수})로 ACT 정책을 1회 실행한다."""
    runner.reset()
    t0 = time.perf_counter()
    step = 0
    period = 1.0 / max(fps, 1.0)

    while time.perf_counter() - t0 < max_duration_s:
        tick = time.perf_counter()
        obs = robot.get_observation()
        joints = joint_positions(obs)
        frames: dict[str, np.ndarray] = {}
        for name in runner.camera_keys:
            src = cameras.get(name)
            if src is None:
                raise RuntimeError(f"카메라 '{name}'가 열려 있지 않음")
            if callable(src):
                frames[name] = src()
            else:
                ok, frame = src.read()
                if not ok:
                    raise RuntimeError(f"카메라 '{name}' 프레임 읽기 실패")
                frames[name] = frame

        goal = runner.predict(joints, frames)
        robot.send_action({f"{m}.pos": goal[m] for m in ALL_MOTORS})
        step += 1

        if on_step is not None and on_step(step, joints, goal, frames) is False:
            break

        wait = period - (time.perf_counter() - tick)
        if wait > 0:
            time.sleep(wait)

    final_joints = joint_positions(robot.get_observation())
    return ACTRolloutResult(
        steps=step,
        elapsed_s=round(time.perf_counter() - t0, 2),
        final_joints=final_joints,
    )
