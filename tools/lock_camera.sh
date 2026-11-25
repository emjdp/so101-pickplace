#!/usr/bin/env bash
# 탑뷰(Logitech C270)와 손목(icSpring, 있을 때) 카메라의 자동 노출 / 자동 화이트밸런스를 끄고 값을 고정한다.
#
# 왜: 자동 기능이 켜져 있으면 팔(흰색 큰 물체)이 화면에 들어올 때마다 밝기와 색이 바뀐다.
#     HSV 범위로 색을 잡는 시스템 A와 영상으로 학습하는 시스템 B(ACT) 모두 이 변화에 흔들린다.
#     USB 를 다시 꽂으면 초기화되므로 세션 시작마다 실행한다.
#
# 사용법:
#   tools/lock_camera.sh          # 적용 후 되읽어서 확인
#   tools/lock_camera.sh --show   # 현재 값만 보기
#
# 값 조정: camera_preview.py 를 띄워 둔 채 아래 값을 바꿔 가며 다시 실행한다.
#   (C270 은 초점이 고정이라 초점 설정은 없다)

set -uo pipefail
DEV="${CAM:-/dev/soarm_cam_top}"
WRIST_DEV="${WRIST_CAM:-/dev/soarm_cam_wrist}"

# ── 탑뷰 C270 ──────────────────────────────────────────────────────
# 노출 단위는 100 µs. 형광등/LED 깜빡임(60 Hz 전원 → 120 Hz)과 맞추려면
# 1/120 s(≈83)의 배수가 좋다: 83, 166, 250, 333
EXPOSURE=83
GAIN=32          # 0-255. 노출을 못 올릴 때만 올린다 (잡음 증가)
WB=4000          # 0-10000 (실제 반응은 2800-6500 근처). 푸르면 올리고 누러면 내린다

# ── 손목 icSpring (연결되어 있을 때만 적용) ──
WRIST_EXPOSURE=120
WRIST_GAIN=15
WRIST_SATURATION=90
WRIST_CONTRAST=38
WRIST_GAMMA=100
WRIST_WB=4300

show() {
    echo "── top ($DEV) ──"
    v4l2-ctl -d "$DEV" -C auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,gain,white_balance_automatic,white_balance_temperature
    if [ -e "$WRIST_DEV" ]; then
        echo "── wrist ($WRIST_DEV) ──"
        v4l2-ctl -d "$WRIST_DEV" -C auto_exposure,exposure_time_absolute,gain,saturation,contrast,gamma,white_balance_automatic,white_balance_temperature
    fi
}

if [ ! -e "$DEV" ]; then
    echo "❌ $DEV 없음 (ls -l /dev/soarm_* , v4l2-ctl --list-devices)" >&2
    exit 1
fi

if [ "${1:-}" = "--show" ]; then
    show
    exit 0
fi

fail=0
expect() {
    local dev=$1 name=$2 want=$3 got
    got=$(v4l2-ctl -d "$dev" -C "$name" 2>/dev/null | awk '{print $2}')
    if [ "$got" = "$want" ]; then echo "  ✅ $name = $got"; else echo "  ❌ $name = $got (원하는 값 $want)"; fail=1; fi
}

echo "── top ($DEV) ──"
# 자동 기능을 먼저 끄고 값을 넣는다 (자동이 켜져 있으면 값 쓰기가 무시된다)
v4l2-ctl -d "$DEV" -c auto_exposure=1 -c exposure_dynamic_framerate=0 -c white_balance_automatic=0
v4l2-ctl -d "$DEV" -c exposure_time_absolute=$EXPOSURE -c gain=$GAIN -c white_balance_temperature=$WB
expect "$DEV" auto_exposure 1
expect "$DEV" exposure_time_absolute $EXPOSURE
expect "$DEV" gain $GAIN
expect "$DEV" white_balance_automatic 0
expect "$DEV" white_balance_temperature $WB

if [ -e "$WRIST_DEV" ]; then
    echo "── wrist ($WRIST_DEV) ──"
    v4l2-ctl -d "$WRIST_DEV" -c auto_exposure=1 -c white_balance_automatic=0
    v4l2-ctl -d "$WRIST_DEV" \
        -c exposure_time_absolute=$WRIST_EXPOSURE \
        -c gain=$WRIST_GAIN \
        -c saturation=$WRIST_SATURATION \
        -c contrast=$WRIST_CONTRAST \
        -c gamma=$WRIST_GAMMA \
        -c white_balance_temperature=$WRIST_WB
    expect "$WRIST_DEV" auto_exposure 1
    expect "$WRIST_DEV" exposure_time_absolute $WRIST_EXPOSURE
    expect "$WRIST_DEV" gain $WRIST_GAIN
    expect "$WRIST_DEV" white_balance_automatic 0
    expect "$WRIST_DEV" white_balance_temperature $WRIST_WB
fi

exit $fail
