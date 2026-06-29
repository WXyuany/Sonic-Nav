#!/usr/bin/env python3
"""Mix two G1 motions: upper body from one, lower body from another."""
import sys, os, numpy as np, tarfile, io, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = "/home/wxy/seed/g1.tar.gz"
OUT_DIR = REPO + "/gear_sonic_deploy/reference/example/mixed_reach_walk"

# Joint indices: 0-11 legs, 12-14 waist, 15-28 arms
LEG_JOINTS = list(range(12))           # 0-11
WAIST_JOINTS = [12, 13, 14]            # 12, 13, 14
ARM_JOINTS = list(range(15, 29))       # 15-28

def get_csv(tar, csv_path):
    """Read a G1 joint_pos CSV from tar."""
    f = tar.extractfile(csv_path)
    return np.loadtxt(f, delimiter=',', skiprows=1)

def write_motion(name, joint_pos, body_pos, body_quat, body_lin, body_ang, joint_vel):
    d = f"{OUT_DIR}_{name}"
    os.makedirs(d, exist_ok=True)
    np.savetxt(f"{d}/joint_pos.csv", joint_pos, delimiter=',',
               header=",".join(f"joint_{i}" for i in range(joint_pos.shape[1])), comments='')
    np.savetxt(f"{d}/body_pos.csv", body_pos, delimiter=',',
               header=",".join(f"body_{i}_{a}" for i in range(body_pos.shape[1]//3) for a in 'xyz'), comments='')
    np.savetxt(f"{d}/body_quat.csv", body_quat, delimiter=',',
               header=",".join(f"body_{i}_{a}" for i in range(body_quat.shape[1]//4) for a in 'wxyz'), comments='')
    np.savetxt(f"{d}/body_lin_vel.csv", body_lin, delimiter=',',
               header=",".join(f"body_{i}_vel_{a}" for i in range(body_lin.shape[1]//3) for a in 'xyz'), comments='')
    np.savetxt(f"{d}/body_ang_vel.csv", body_ang, delimiter=',',
               header=",".join(f"body_{i}_angvel_{a}" for i in range(body_ang.shape[1]//3) for a in 'xyz'), comments='')
    np.savetxt(f"{d}/joint_vel.csv", joint_vel, delimiter=',',
               header=",".join(f"joint_vel_{i}" for i in range(joint_vel.shape[1])), comments='')
    with open(f"{d}/metadata.txt", "w") as f:
        f.write(f"mixed motion: {name}")
    print(f"  Created: {d}")

def main():
    walk_csv = None
    reach_csv = None
    args = sys.argv[1:]
    
    tar = tarfile.open(SEED, 'r:gz')
    
    # Search for walking + reaching motions
    for member in tar.getmembers():
        if not member.name.endswith('.csv'): continue
        name = os.path.basename(member.name).lower()
        if walk_csv is None and 'walk' in name and 'forward' in name and '_M' not in name:
            walk_csv = member.name
            print(f"Walk: {member.name}")
        if reach_csv is None and ('reach' in name or 'grab' in name or 'pick' in name) and 'walk' not in name and '_M' not in name:
            reach_csv = member.name
            print(f"Reach: {member.name}")
        if walk_csv and reach_csv: break

    if not walk_csv or not reach_csv:
        print("Could not find both motions")
        tar.close()
        return

    # Read joint_pos from both
    walk_jp = get_csv(tar, walk_csv)
    reach_jp = get_csv(tar, reach_csv)

    # Match lengths (use shorter)
    L = min(len(walk_jp), len(reach_jp))
    walk_jp = walk_jp[:L]
    reach_jp = reach_jp[:L]

    # Mix: walk body + legs, reach arms
    mixed = walk_jp.copy()
    mixed[:, ARM_JOINTS] = reach_jp[:, ARM_JOINTS]

    # Generate dummy body data (walk in place)
    body_pos = np.tile(np.array([[0,0,0.79]*14]), (L,1))
    body_quat = np.tile(np.array([[1,0,0,0]*14]), (L,1))
    body_lin = np.zeros((L, 42))
    body_ang = np.zeros((L, 42))
    joint_vel = np.zeros((L, 29))

    write_motion("reach_walk", mixed, body_pos, body_quat, body_lin, body_ang, joint_vel)

    tar.close()
    print("\nDone! Add to reference/example/ and restart deploy.")

if __name__ == '__main__':
    main()
