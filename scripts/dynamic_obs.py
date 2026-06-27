#!/usr/bin/env python3
import numpy as np, mujoco, time, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml = REPO + '/gear_sonic/data/robot_model/model_data/g1/scene_dynamic.xml'
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)

actuator_map = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        actuator_map[name] = i

t0 = time.time()
while True:
    try:
        qpos = np.load('/tmp/sonic_qpos.npy')
        data.qpos[:len(qpos)] = qpos
    except:
        time.sleep(0.01)
        continue

    mujoco.mj_step(model, data)
    t = time.time() - t0

    if 'a1' in actuator_map:
        data.ctrl[actuator_map['a1']] = 2.5 * np.sin(t * 0.8)
    if 'a2' in actuator_map:
        data.ctrl[actuator_map['a2']] = 2.5 * np.cos(t * 0.6)
    if 'a3' in actuator_map:
        data.ctrl[actuator_map['a3']] = 0.5 * np.sin(t * 0.3)

    np.save('/tmp/sonic_qpos.npy', data.qpos.copy())
    time.sleep(max(0, 0.002 - (time.time() - t0 - t + time.time() - t0)))
