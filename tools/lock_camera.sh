#!/usr/bin/env bash
# C270 의 자동 노출 / 자동 화이트밸런스를 끄고 값을 고정한다.
#
# 왜: 자동 기능이 켜져 있으면 팔(흰색 큰 물체)이 화면에 들어올 때마다 밝기와 색이 바뀐다.
#     HSV 범위로 색을 잡는 검출은 이 변화에 그대로 흔들린다.
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

# 노출 단위는 100 µs. 형광등/LED 깜빡임(60 Hz 전원 → 120 Hz)과 맞추려면
# 1/120 s(≈83)의 배수가 좋다: 83, 166, 250, 333
EXPOSURE=166
GAIN=64          # 0-255. 노출을 못 올릴 때만 올린다 (잡음 증가)
WB=4000          # 0-10000 (실제 반응은 2800-6500 근처). 푸르면 올리고 누러면 내린다

show() {
    v4l2-ctl -d "$DEV" -C auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,gain,white_balance_automatic,white_balance_temperature
}

if [ ! -e "$DEV" ]; then
    echo "❌ $DEV 없음 (ls -l /dev/soarm_* , v4l2-ctl --list-devices)" >&2
    exit 1
fi

if [ "${1:-}" = "--show" ]; then
    show
    exit 0
fi

# 자동 기능을 먼저 끄고 값을 넣는다 (자동이 켜져 있으면 값 쓰기가 무시된다)
v4l2-ctl -d "$DEV" -c auto_exposure=1 -c exposure_dynamic_framerate=0 -c white_balance_automatic=0
v4l2-ctl -d "$DEV" -c exposure_time_absolute=$EXPOSURE -c gain=$GAIN -c white_balance_temperature=$WB

fail=0
expect() {
    local got
    got=$(v4l2-ctl -d "$DEV" -C "$1" | awk '{print $2}')
    if [ "$got" = "$2" ]; then echo "✅ $1 = $got"; else echo "❌ $1 = $got (원하는 값 $2)"; fail=1; fi
}
expect auto_exposure 1
expect exposure_time_absolute $EXPOSURE
expect gain $GAIN
expect white_balance_automatic 0
expect white_balance_temperature $WB
exit $fail
