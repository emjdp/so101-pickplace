# so101-pickplace

탑뷰 카메라 기반 SO-101 픽앤플레이스. 기구학 기반 제어(시스템 A)와 ACT 모방학습(시스템 B)을 같은 조건에서 비교한다.
소프트웨어융합개론 텀프로젝트 1팀.

공부가 목적이라 핵심 계산은 **빈칸(★)으로 남겨 두고 직접 채운다.** 장치 연결·화면 표시 같은 주변 코드는 미리 만들어 두었다.
진행 순서와 개념 설명은 팀 노트(`Term-Project/개발/`)에 있다.

## 구조

| 경로 | 내용 | 상태 |
|---|---|---|
| `config.py` | 포트, 카메라, ID, 파일 위치 | 완성 |
| `so101/transforms.py` | 회전행렬, 동차변환 | ★ 빈칸 |
| `so101/fk.py` | LeRobot 각도 → URDF 각도, 순기구학 | ★ 빈칸 |
| `so101/detect.py` | HSV 마스크, 블록 중심·각도 | ★ 빈칸 |
| `so101/calib.py` | 체커보드 3D 점, 재투영 오차 | ★ 빈칸 |
| `so101/urdf.py`, `robot.py`, `camera.py` | URDF 읽기, LeRobot 연결, 카메라 열기 | 완성 (읽어 볼 것) |
| `scripts/` | 실습 실행 스크립트. 빈칸 함수를 불러 쓴다 | 완성 |
| `tools/` | 점검, 노출 고정, 프리뷰, 체커보드, HSV 조정 | 완성 |
| `tests/` | 빈칸 채점용 테스트 | 완성 |
| `calib/` | LeRobot 캘리브레이션, 카메라 내부 파라미터, HSV 범위 (커밋한다) | 실습 결과 |
| `data/` | 측정 CSV, 사진 | 실습 결과 |

## 처음 한 번

```bash
conda activate lerobot
pip uninstall -y opencv-python-headless && pip install opencv-python==4.13.0.92   # 창 띄우기용
pip install pytest yourdfpy matplotlib
git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100      # URDF
sudo cp udev/99-soarm.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger
python tools/check_setup.py
```

- ROS 2 가 켜져 있으면 끄고 시작한다 (`echo $ROS_DISTRO` 가 비어 있어야 함).
- lerobot 을 `pip install -e` 로 다시 설치하면 headless OpenCV 가 되돌아온다. 그때는 첫 줄의 OpenCV 교체를 다시 한다.

## 실행 순서 (1~4주차)

```bash
# 모터 캘리브레이션 (결과: calib/lerobot/)
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/soarm_follower --robot.id=tp_follower --robot.calibration_dir=calib/lerobot/follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/soarm_leader --teleop.id=tp_leader --teleop.calibration_dir=calib/lerobot/leader

python scripts/read_joints.py                 # 관절값 보기 (리더)
python scripts/move_joint.py shoulder_pan --delta 15

pytest tests/test_transforms.py tests/test_fk.py
python scripts/fk_try.py 0 0 0 0 0 --plot     # 로봇 없이 FK
python scripts/go_pose.py 0 0 0 90 0          # 영점 확인
python scripts/fk_live.py                     # FK 실측 비교

tools/lock_camera.sh && python tools/camera_preview.py
pytest tests/test_detect.py && python tools/hsv_tuner.py blue && python scripts/detect_live.py blue
python tools/make_checkerboard.py && python scripts/calib_capture.py
pytest tests/test_calib.py && python scripts/calib_intrinsics.py --square-mm <잰 값>
```

## 장치

| 장치 | 링크 | 비고 |
|---|---|---|
| 팔로워 | `/dev/soarm_follower` | CH343 시리얼 5B7B015254 |
| 리더 | `/dev/soarm_leader` | CH343 시리얼 5B7B014404 |
| 탑뷰 C270 | `/dev/soarm_cam_top` | 640x480 YUYV, 초점 고정 |

## AI 사용 표시

과제 규정에 따라 AI 가 만든 코드는 출처를 남긴다. 이 저장소의 초기 뼈대(완성 도구, 빈칸 틀, 테스트)는
Claude Opus 5.5 로 만들었다 [AI 생성 도구: Claude Opus 5.5, 프롬프트: '텀프로젝트 1~4주차를 팀원과 공부하며 진행할 수 있게 준비'].
빈칸은 팀원이 직접 채운다.
