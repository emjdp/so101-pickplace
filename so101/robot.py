"""LeRobot 팔로워/리더 연결 도우미. (실습 대상 아님. 코드 읽기는 권장)"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

import config

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_MOTORS = ARM_JOINTS + ["gripper"]


def connect_follower(max_relative_target=10.0):
    """팔로워에 연결한다.

    max_relative_target: 한 번의 send_action 에서 관절이 움직일 수 있는 최대 각도(deg).
    목표가 현재 위치보다 이보다 멀면 LeRobot 이 잘라낸다. 잘못된 명령으로 팔이 튀는 것을 막는다.
    """
    cfg = SO101FollowerConfig(
        port=config.FOLLOWER_PORT,
        id=config.FOLLOWER_ID,
        calibration_dir=config.FOLLOWER_CALIB_DIR,
        max_relative_target=max_relative_target,
        use_degrees=True,
    )
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def connect_leader():
    cfg = SO101LeaderConfig(
        port=config.LEADER_PORT,
        id=config.LEADER_ID,
        calibration_dir=config.LEADER_CALIB_DIR,
        use_degrees=True,
    )
    leader = SO101Leader(cfg)
    leader.connect()
    return leader


def joint_positions(obs):
    """get_observation() / get_action() 결과에서 관절값만 {이름: deg} 로 뽑는다."""
    return {m: float(obs[f"{m}.pos"]) for m in ALL_MOTORS}


def hold_current_pose(robot):
    """현재 위치를 목표로 다시 써서 그 자리에 멈춰 있게 한다."""
    obs = robot.get_observation()
    robot.send_action({f"{m}.pos": obs[f"{m}.pos"] for m in ALL_MOTORS})
    return joint_positions(obs)
