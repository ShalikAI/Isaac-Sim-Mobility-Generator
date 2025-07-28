#!/usr/bin/env python3
import os, sys, glob
import numpy as np
from scipy.spatial.transform import Rotation as R


def inspect(folder, num=5):
    files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    for i, fn in enumerate(files[:num]):
        st = np.load(fn, allow_pickle=True).item()

        # --- robot pelvis pose ---
        p_pos = st['robot.position']    # [x,y,z]
        qw, qx, qy, qz = st['robot.orientation']
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

        print(f"[{i}] Pelvis")
        print(f"  Position: (x={p_pos[0]:.3f}, y={p_pos[1]:.3f}, z={p_pos[2]:.3f})")
        print(f"  Quaternion: qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}")
        print(f"  Euler:     roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°\n")

        # --- front‑left camera pose ---
        l_pos = st['robot.front_camera.left.position']
        lqw, lqx, lqy, lqz = st['robot.front_camera.left.orientation']
        lrot = R.from_quat([lqx, lqy, lqz, lqw])
        lroll, lpitch, lyaw = lrot.as_euler('xyz', degrees=True)

        print(f"[{i}] Front‑Left Camera")
        print(f"  Position: (x={l_pos[0]:.3f}, y={l_pos[1]:.3f}, z={l_pos[2]:.3f})")
        print(f"  Quaternion: qw={lqw:.3f}, qx={lqx:.3f}, qy={lqy:.3f}, qz={lqz:.3f}")
        print(f"  Euler:     roll={lroll:.1f}°, pitch={lpitch:.1f}°, yaw={lyaw:.1f}°")
        print('-' * 50)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: inspect_poses.py /path/to/state_folder [num]")
        sys.exit(1)
    folder = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    inspect(folder, n)
