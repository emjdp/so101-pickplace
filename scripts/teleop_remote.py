"""리더암(노트북)으로 Raspberry Pi에 연결한 팔로워를 조종한다. 관절값만 UDP 로 보낸다.

rpi5 (팔로워 쪽, 먼저 띄운다)
  python scripts/teleop_remote.py follower
노트북 (리더 쪽)
  python scripts/teleop_remote.py leader                 # 기본 대상: config.FOLLOWER_HOST

lerobot-teleoperate 와 같은 방식이다: 리더가 읽은 값(LeRobot deg, 그리퍼 0~100)을 팔로워 목표로 그대로 쓴다.

안전장치 (팔로워 쪽)
- 처음 연결되거나 신호가 끊겼다 돌아오면 목표를 --sync-speed(기본 30°/s)로 천천히 리더 자세까지 옮긴다.
  따라잡은 뒤에는 --max-speed(기본 180°/s)까지만 따라간다.
- --timeout(기본 0.5 s) 동안 패킷이 안 오면 명령을 멈춘다. 모터는 마지막 목표에서 버틴다.
- 순서가 뒤바뀐 옛 패킷은 버린다. 리더가 둘이면 먼저 붙은 쪽만 따른다.
- 종료할 때 팔로워 토크는 유지한다. 토크를 끄려면 `--release`를 쓴다.
"""

import argparse
import json
import math
import random
import select
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from so101.robot import ALL_MOTORS, connect_follower, connect_leader, hold_current_pose, joint_positions  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_all(sock):
    """소켓에 쌓인 패킷을 모두 꺼낸다. 깨진 것은 버린다."""
    out = []
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except BlockingIOError:
            return out
        try:
            out.append((json.loads(data), addr))
        except ValueError:
            pass


def parse_command(msg):
    """리더 패킷 → (sid, seq, {관절: 값}). 형식이 틀리거나 값이 이상하면 None."""
    try:
        sid, seq = int(msg["sid"]), int(msg["seq"])
        pos = {m: float(msg["pos"][m]) for m in ALL_MOTORS}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in pos.values()):
        return None
    return sid, seq, pos


def step_toward(goal, target, max_step):
    """목표를 관절마다 max_step 이내로 target 쪽으로 옮긴다. 다 도착했으면 True 도 돌려준다."""
    out, arrived = {}, True
    for m in ALL_MOTORS:
        d = target[m] - goal[m]
        if abs(d) > max_step:
            d = math.copysign(max_step, d)
            arrived = False
        out[m] = goal[m] + d
    return out, arrived


def run_follower(args):
    robot = connect_follower(max_relative_target=None)  # 속도 제한은 아래에서 직접 한다
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp_port))
    sock.setblocking(False)

    goal = hold_current_pose(robot)
    log(f"팔로워 {config.FOLLOWER_PORT} 연결. UDP {args.udp_port} 에서 리더를 기다린다")

    sid = last_seq = target = leader_addr = None
    last_rx = last_cmd = last_fb = -math.inf
    syncing, lost = True, False
    try:
        while True:
            select.select([sock], [], [], 1 / args.hz)
            now = time.monotonic()

            for msg, addr in read_all(sock):
                cmd = parse_command(msg)
                if cmd is None:
                    continue
                c_sid, c_seq, c_pos = cmd
                if c_sid != sid:
                    if now - last_rx < args.timeout:
                        continue  # 지금 리더가 살아 있는 동안 다른 리더는 무시
                    sid, last_seq = c_sid, -1
                    goal = joint_positions(robot.get_observation())
                    syncing, lost = True, False
                    worst = max(ALL_MOTORS, key=lambda m: abs(c_pos[m] - goal[m]))
                    log(f"리더 연결 {addr[0]}:{addr[1]}, 최대 차이 {c_pos[worst] - goal[worst]:+.1f} ({worst})"
                        f" → {args.sync_speed:.0f}°/s 로 리더 자세까지 맞춘다")
                if c_seq <= last_seq:
                    continue
                last_seq, target, leader_addr, last_rx = c_seq, c_pos, addr, now

            if target is None:
                continue
            if now - last_rx > args.timeout:
                if not lost:
                    log(f"리더 신호가 {args.timeout} s 끊김 → 제자리 유지")
                    lost = True
                continue
            if lost:
                goal = joint_positions(robot.get_observation())
                syncing, lost = True, False
                log("리더 신호 복구 → 다시 천천히 맞춘다")
                last_cmd = now

            dt = min(now - last_cmd, 0.1)
            last_cmd = now
            speed = args.sync_speed if syncing else args.max_speed
            goal, arrived = step_toward(goal, target, speed * dt)
            robot.send_action({f"{m}.pos": v for m, v in goal.items()})
            if syncing and arrived:
                syncing = False
                log(f"리더 자세에 맞춤 → 추종 (최대 {args.max_speed:.0f}°/s)")

            if now - last_fb >= 0.1:
                last_fb = now
                present = joint_positions(robot.get_observation())
                fb = {"seq": last_seq, "syncing": syncing, "present": present}
                sock.sendto(json.dumps(fb).encode(), leader_addr)
    except KeyboardInterrupt:
        pass
    finally:
        robot.config.disable_torque_on_disconnect = args.release
        robot.disconnect()
        sock.close()
        log("팔로워 종료 (토크 " + ("끔" if args.release else "유지") + ")")


def run_leader(args):
    addr = socket.getaddrinfo(args.host, args.udp_port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    leader = connect_leader()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sid = random.getrandbits(31)
    log(f"리더 {config.LEADER_PORT} 연결. {args.host} ({addr[0]}:{addr[1]}) 로 {args.hz:.0f} Hz 송신. Ctrl+C 로 끝")

    seq, sent_at = 0, {}
    fb, fb_t, rtt = None, -math.inf, float("nan")
    next_t, next_print = time.monotonic(), 0.0
    try:
        while True:
            pos = joint_positions(leader.get_action())
            seq += 1
            sent_at[seq] = time.monotonic()
            sent_at.pop(seq - 200, None)
            sock.sendto(json.dumps({"sid": sid, "seq": seq, "pos": pos}).encode(), addr)

            for msg, _ in read_all(sock):
                if isinstance(msg, dict) and "present" in msg:
                    fb, fb_t = msg, time.monotonic()
                    if msg.get("seq") in sent_at:
                        rtt = fb_t - sent_at[msg["seq"]]

            now = time.monotonic()
            if now >= next_print:
                next_print = now + 0.2
                if now - fb_t > 1.0:
                    status = f"팔로워 응답 없음 ({addr[0]}:{addr[1]}, 팔로워 쪽을 먼저 띄웠는지 확인)"
                else:
                    err = {m: abs(pos[m] - fb["present"][m]) for m in ALL_MOTORS if m in fb["present"]}
                    worst = max(err, key=err.get)
                    state = "맞추는 중" if fb.get("syncing") else "추종"
                    status = f"팔로워 {state} | 왕복 {rtt * 1000:4.0f} ms | 최대 차이 {err[worst]:5.1f} ({worst})"
                print(f"\r{status}\033[K", end="", flush=True)

            next_t += 1 / args.hz
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        print()
    finally:
        leader.disconnect()
        sock.close()
        log("리더 종료. 팔로워는 마지막 자세에서 버틴다")


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
sub = ap.add_subparsers(dest="role", required=True)
f = sub.add_parser("follower", help="팔로워가 달린 컴퓨터(rpi5)에서")
f.add_argument("--max-speed", type=float, default=180.0, help="추종 중 최대 속도 [deg/s]")
f.add_argument("--sync-speed", type=float, default=30.0, help="처음 맞출 때 속도 [deg/s]")
f.add_argument("--timeout", type=float, default=0.5, help="이만큼 패킷이 없으면 멈춤 [s]")
f.add_argument("--release", action="store_true", help="끝낼 때 토크를 끈다 (팔이 처진다)")
f.set_defaults(run=run_follower)
le = sub.add_parser("leader", help="리더가 달린 컴퓨터(노트북)에서")
le.add_argument("--host", default=config.FOLLOWER_HOST)
le.set_defaults(run=run_leader)
for p in (f, le):
    p.add_argument("--udp-port", type=int, default=config.TELEOP_UDP_PORT)
    p.add_argument("--hz", type=float, default=50.0)
args = ap.parse_args()
args.run(args)
