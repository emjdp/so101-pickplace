# SO-101 픽앤플레이스

소프트웨어융합개론 텀프로젝트 1팀 (김재현, 박성호).

SO-101 로봇팔로 블록을 집어 상자에 넣는다. 시스템 A는 탑뷰 영상에서 블록을 찾아 역기구학으로 관절 목표를 계산한다. 시스템 B는 리더암으로 녹화한 시연을 ACT로 학습하고, 탑뷰 영상과 현재 관절 상태로 다음 동작을 예측한다.

## 환경 준비

Ubuntu 24.04, Python 3.12, LeRobot 0.6.1 (`d36f429`) 환경에서 확인했다. 기존 LeRobot 환경이 있으면 그대로 쓰면 된다. 새 환경에서는 이 저장소의 루트에서 다음 명령을 실행한다.

```bash
conda create -n lerobot python=3.12
conda activate lerobot
python -m pip install 'lerobot[core_scripts,feetech] @ git+https://github.com/huggingface/lerobot.git@d36f429a304d87ef0a4ffc6c17ce0feb28b30e0d'
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python==4.13.0.92 pytest yourdfpy matplotlib
git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
git -C third_party/SO-ARM100 checkout a758567c3978dfeefe282ede0500085a48fe8f78
python -m pytest -q
```

카메라 미리보기나 HSV 조정 창을 띄우려면 일반 OpenCV 빌드가 필요하다. 실기에서는 ROS 2를 불러오지 않은 터미널을 쓰고, 장치 시리얼이 다르면 `udev/99-soarm.rules`를 고친 뒤 적용한다. 모터나 카메라 장착 위치를 바꿨으면 기존 보정값도 다시 확인한다.

```bash
sudo install -m 644 udev/99-soarm.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
python tools/check_setup.py
```

## 시스템 A · 기구학 제어

HSV로 블록의 중심과 방향을 찾는다. 호모그래피로 로봇 기준 위치를 구한 뒤 접근, 하강, 파지, 이송, 놓기, 복귀 순서로 움직인다. 모터와 카메라 보정값은 `calib/`에 있다.

```bash
python scripts/run_system_a.py blue --dry-run  # 합성 영상과 가상 팔로워
tools/lock_camera.sh
python tools/hsv_tuner.py blue                 # s 키로 HSV 범위 저장
python scripts/detect_live.py blue --undistort
python scripts/calib_homography.py --verify
python scripts/run_system_a.py blue
```

카메라 내부 보정을 다시 하려면 [체커보드 PDF](output/pdf/checkerboard_9x6_25mm.pdf)를 실제 크기 100%로 인쇄하고, 잰 칸 크기를 `scripts/calib_intrinsics.py --square-mm`에 넘긴다. 카메라 외부 자세에서 블록 윗면(25 mm) 기준 호모그래피를 다시 만들려면 `python scripts/calib_homography.py --from-extrinsics --plane-z 0.025`를 실행한다. 외부 자세로 계산한 호모그래피의 `rms_mm: 0.0`은 실측 위치 오차로 쓰지 않는다. 실기에서는 별도로 찍은 기준점의 위치와 투영 좌표를 비교한다.

## 시스템 B · ACT

본 비교용 시연은 탑뷰만 써서 약 50개를 모은다. 손목 카메라를 추가할 때는 별도 데이터셋을 사용한다. 녹화 세션을 다시 시작해도 같은 데이터셋에는 같은 카메라 구성을 유지한다.

```bash
# 터미널 1
tools/lock_camera_live.sh --watch

# 터미널 2
tools/record_act.sh 10 --top-only
tools/record_act.sh --pack  # 시연 수집을 마친 뒤 실행
```

녹화 중 오른쪽 화살표로 현재 에피소드를 저장하고, 왼쪽 화살표로 해당 에피소드를 다시 찍는다. `Esc`로 세션을 끝낸다.

`data/so101_pickplace_dataset.tar.gz`를 Seraph의 `/data/$USER/datasets/`에 올린다. LeRobot 학습 환경을 준비하고, 로그 디렉터리를 만든 뒤 `seraph/train_act.sbatch`를 제출한다.

```bash
mkdir -p "/data/$USER/logs"
WANDB_MODE=offline sbatch seraph/train_act.sbatch
```

학습을 마치면 `checkpoints/last/pretrained_model`을 로컬의 `outputs/act_pickplace/pretrained_model`로 내려받는다. 먼저 가상 관절값과 영상으로 추론을 확인한 뒤 실기에서 실행한다.

```bash
python scripts/run_system_b.py --dry-run
tools/lock_camera.sh
python scripts/run_system_b.py
```

## A/B 비교 절차

기본, 새 위치, 색 변경, 조명 변경 조건에서 방식별 20회씩 총 160회를 계획했다. 조건별로 블록 시작 위치와 방향 20개를 정해 A와 B에 똑같이 적용하고, 순서를 번갈아 가며 시행한다. 매번 팔의 시작 자세와 상자 위치를 맞춘다. 색 변경 조건에서는 블록 색만 바꾸고 HSV 범위와 정책을 그대로 둔다.

블록 전체가 제한 시간 안에 상자에 들어가 2초 이상 머무르면 성공이다. 현재 실행 스크립트는 서로 다른 구간의 시간을 기록하므로, 본 평가에서는 두 방식 모두 첫 명령부터 성공 판정까지 같은 기준으로 잰다. 예비 실험 뒤 제한 시간을 확정하고, 시행별 위치·방향과 실패 원인을 함께 남긴다.

```bash
python scripts/eval_compare.py interactive --system A --condition base --count 20
python scripts/eval_compare.py summary --out data/experiments/summary.md
```

`--dry-run`에서는 `--log`를 빼고 실행한다. 두 옵션을 함께 쓰면 가상 시행도 기본 실험 CSV에 들어간다. 현재 확인한 범위는 테스트 50개, 시스템 A의 합성 영상 실행, 기존 ACT 체크포인트를 사용한 시스템 B의 CPU 추론이다. 새 블록 시연으로 학습한 정책과 두 방식의 실기 비교 결과는 확인이 필요하다.

## 원격 텔레오퍼레이션

팔로워를 Raspberry Pi에 연결한 경우, 팔로워 쪽부터 실행한다. 리더는 관절값을 UDP 5560번 포트로 보낸다.

```bash
# Raspberry Pi
python scripts/teleop_remote.py follower

# 리더암이 연결된 노트북
python scripts/teleop_remote.py leader  # 대상은 config.FOLLOWER_HOST에서 설정
```
