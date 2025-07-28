#!/usr/bin/env python3
import os, sys, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def inspect(folder, num=5):
    files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    for i, fn in enumerate(files[:num]):
        st  = np.load(fn, allow_pickle=True).item()

        pos = st['robot.position']      # [x,y,z]
        qw, qx, qy, qz = st['robot.orientation']  # USD gives [qw, qx, qy, qz]

        # reorder to SciPy's [x, y, z, w]
        rot = R.from_quat([qx, qy, qz, qw])

        # 'xyz' → [roll, pitch, yaw]
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

        print(f"Frame {i:02d}:")
        print(f"  raw pos   = {pos}")
        print(f"  raw quat  = [qw={qw:.3f}, qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}]")
        print(f"  → roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: inspect_poses.py /path/to/state/common [num]")
        sys.exit(1)
    folder = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    inspect(folder, n)
