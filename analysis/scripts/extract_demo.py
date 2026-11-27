"""시연 데이터셋에서 에피소드별 첫 장면(탑뷰)과 표본 프레임을 뽑고, 관절 기록을 정리한다."""
import glob, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

DS = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "first").mkdir(exist_ok=True); (OUT / "sample").mkdir(exist_ok=True)
ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DS / "meta/episodes/chunk-000/*.parquet")))])
ep = ep.sort_values("episode_index").reset_index(drop=True)
key = "videos/observation.images.top"
rows = []
for _, e in ep.iterrows():
    vid = DS / f"videos/observation.images.top/chunk-{int(e[key+'/chunk_index']):03d}/file-{int(e[key+'/file_index']):03d}.mp4"
    t0, t1 = float(e[key + "/from_timestamp"]), float(e[key + "/to_timestamp"])
    i = int(e["episode_index"])
    f = OUT / "first" / f"ep{i:02d}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0 + 0.02:.3f}", "-i", str(vid), "-frames:v", "1", str(f)], check=True)
    # 에피소드 전체에서 1초 간격 표본
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(vid),
                    "-vf", "fps=1", str(OUT / "sample" / f"ep{i:02d}_%02d.png")], check=True)
    rows.append({"episode": i, "length": int(e["length"]), "video": vid.name, "t0": t0, "t1": t1})
pd.DataFrame(rows).to_csv(OUT / "episodes.csv", index=False)
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DS / "data/chunk-000/*.parquet")))])
st = np.stack(d["observation.state"].to_numpy()); ac = np.stack(d["action"].to_numpy())
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
df = pd.DataFrame({"episode": d["episode_index"].to_numpy(), "frame": d["frame_index"].to_numpy(), "t": d["timestamp"].to_numpy()})
for k, n in enumerate(names):
    df["state_" + n] = st[:, k]; df["action_" + n] = ac[:, k]
df.to_csv(OUT / "joints.csv", index=False)
print(len(rows), "episodes;", len(list((OUT / "sample").glob("*.png"))), "sample frames")
