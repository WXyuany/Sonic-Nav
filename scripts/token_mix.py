#!/usr/bin/env python3
import os, sys, time, msgpack, zmq, numpy as np, onnxruntime as ort
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_DIR = f"{REPO}/gear_sonic_deploy/reference/example"
ENCODER = f"{REPO}/gear_sonic_deploy/policy/release/model_encoder.onnx"
motions = [m for m in sorted(os.listdir(MOTION_DIR)) if os.path.isdir(f"{MOTION_DIR}/{m}")]
for i, m in enumerate(motions): print(f"  [{i}] {m}")
a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (0, 3)
m1, m2 = motions[a], motions[b]; print(f"\nMixing [{a}] {m1} + [{b}] {m2}")
jp1 = np.loadtxt(f"{MOTION_DIR}/{m1}/joint_pos.csv", delimiter=',', skiprows=1)
jp2 = np.loadtxt(f"{MOTION_DIR}/{m2}/joint_pos.csv", delimiter=',', skiprows=1)
L = min(len(jp1), len(jp2))
session = ort.InferenceSession(ENCODER)
enc_in = session.get_inputs()[0].name; enc_out = session.get_outputs()[0].name
def get_token(jp, t):
    obs = np.zeros(1762, dtype=np.float32)
    for f in range(min(10, t//5+1)):
        src = max(0, t-(9-f)*5)
        for j in range(29): obs[4+f*29+j] = jp[src, j]
    obs[4+290+290+10+1+6+60] = 1.0
    return session.run([enc_out], {enc_in: obs.reshape(1,-1)})[0][0]
t1 = get_token(jp1, min(100, L-1)); t2 = get_token(jp2, min(100, L-1))
print(f"Token norms: {np.linalg.norm(t1):.3f}, {np.linalg.norm(t2):.3f}")
ctx = zmq.Context(); sock = ctx.socket(zmq.PUB); sock.bind("tcp://*:5556")
print("Streaming ZMQ 5556. Ctrl+C to stop.")
frame = 0
while True:
    alpha = 0.5 + 0.3 * np.sin(frame * 0.03)
    mixed = alpha * t1 + (1-alpha) * t2
    sock.send_multipart([b"pose", msgpack.packb({"token_state":mixed.tolist(),"frame_index":frame}, use_bin_type=True)])
    frame += 1; time.sleep(0.02)
