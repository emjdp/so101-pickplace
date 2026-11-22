#!/usr/bin/env bash
# LeRobot이 카메라를 열면 노출과 화이트밸런스를 고정한다.
#
# 왜 필요한가:
#   Logitech 웹캠은 스트림이 시작되는 순간 펌웨어가 자동 노출/AWB로 되돌아갈 수 있다.
#   카메라가 열린 직후 스트리밍 상태에서 lock_camera.sh 를 적용하면 고정값이 유지된다.
#
# 사용법 — 터미널 2개를 쓴다:
#   [터미널 1] tools/lock_camera_live.sh        # 먼저 띄워두고 대기시킨다
#   [터미널 2] tools/record_act.sh              # 카메라가 열리면 자동 적용됨
#
#   tools/lock_camera_live.sh --watch           # 세션 내내 15초마다 재확인
#
# 종료: Ctrl+C

set -uo pipefail

cd "$(dirname "$0")/.."

TOP_DEV="/dev/soarm_cam_top"
WRIST_DEV="/dev/soarm_cam_wrist"
WAIT_TIMEOUT=600
SETTLE=2
VERIFY_AFTER=5
WATCH_INTERVAL=15

WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1

active_cams() {
    local list=()
    [ -e "$TOP_DEV" ] && list+=("$TOP_DEV")
    [ -e "$WRIST_DEV" ] && list+=("$WRIST_DEV")
    echo "${list[@]}"
}

snapshot() {
    for c in auto_exposure exposure_time_absolute white_balance_automatic white_balance_temperature; do
        printf '%s=%s ' "$c" \
            "$(v4l2-ctl -d "$TOP_DEV" -C "$c" 2>/dev/null | awk '{print $2}')"
    done
}

wait_for_stream() {
    local cams
    cams=$(active_cams)
    if [ -z "$cams" ]; then
        echo "❌ 연결된 카메라가 없다 (/dev/soarm_cam_top)" >&2
        return 1
    fi
    echo "⏳ 카메라가 열리기를 기다리는 중... 다른 터미널에서 녹화/실행 스크립트를 띄우세요."
    echo "   (Ctrl+C 로 취소)"
    for _ in $(seq 1 $((WAIT_TIMEOUT * 2))); do
        if fuser -s $cams 2>/dev/null; then
            echo "📷 카메라 열림 감지."
            return 0
        fi
        sleep 0.5
    done
    echo "❌ ${WAIT_TIMEOUT}초 안에 카메라가 열리지 않았다." >&2
    return 1
}

apply_once() {
    sleep "$SETTLE"
    echo "── 적용 중 (스트리밍 상태) ──"
    if tools/lock_camera.sh; then
        :
    else
        echo "⚠  일부 항목 실패. 위 ❌ 확인." >&2
    fi

    local before after
    before=$(snapshot)
    sleep "$VERIFY_AFTER"
    after=$(snapshot)

    echo
    if [ "$before" = "$after" ]; then
        echo "✅ ${VERIFY_AFTER}초 뒤에도 값이 그대로다. 고정 성공."
        echo "   $after"
    else
        echo "❌ 값이 계속 흘러가고 있다. 녹화를 멈추고 확인할 것." >&2
        echo "   적용 직후: $before" >&2
        echo "   ${VERIFY_AFTER}초 뒤: $after" >&2
        return 1
    fi
}

wait_for_stream || exit 1
apply_once

if [ "$WATCH" -eq 1 ]; then
    cams=$(active_cams)
    echo
    echo "👁  --watch: ${WATCH_INTERVAL}초마다 재확인한다. Ctrl+C 로 종료."
    while fuser -s $cams 2>/dev/null; do
        sleep "$WATCH_INTERVAL"
        drift_before=$(snapshot)
        tools/lock_camera.sh >/dev/null 2>&1
        drift_after=$(snapshot)
        if [ "$drift_before" != "$drift_after" ]; then
            echo "⚠  $(date +%H:%M:%S) 값이 흘러서 재적용함"
            echo "     전: $drift_before"
            echo "     후: $drift_after"
        fi
    done
    echo "카메라가 닫혔다. 종료."
fi
