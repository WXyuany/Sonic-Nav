"""Fast MuJoCo skill-level curriculum for hybrid world-model PPO.

This is intentionally not a replacement for the SONIC WBC simulator.  It is a
parallelizable contact curriculum that learns bounded skill residuals before
they are evaluated through the full ROS/WBC stack.  The action/observation
contract is identical to ``HybridRecurrentActorCritic``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .hybrid_ppo import CONTEXT_DIM, ENTITY_DIM


_XML = """
<mujoco model="sonic_skill_curriculum">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 .1" rgba=".25 .25 .25 1"/>
    <geom name="table" type="box" pos=".65 0 .76" size=".65 .55 .04" rgba=".45 .32 .18 1"/>
    <body name="ball" pos=".55 0 .84"><freejoint/><geom name="ball_geom" type="sphere" size=".045" mass=".08" friction="1.2 .01 .001"/></body>
    <body name="hand" mocap="true" pos=".25 0 .95"><geom name="hand_geom" type="sphere" size=".075" mass="1" rgba=".2 .5 .9 1"/></body>
    <site name="target" pos=".75 .24 .84" size=".10" type="cylinder" rgba=".2 .8 .3 .35"/>
  </worldbody>
</mujoco>
"""


@dataclass
class SkillInfo:
    success: np.ndarray
    dropped: np.ndarray
    contact: np.ndarray
    lifted: np.ndarray
    collision: np.ndarray


class MujocoSkillLevelEnv:
    """Vectorized independent MuJoCo worlds with stage-level actions.

    Continuous action: base x/y/yaw residual, wrist pitch, contact x/z,
    close ratio, and lift height.  The categorical recovery action is shared
    with the deployed policy: continue/reobserve/micro-adjust/replan/abort.
    """

    def __init__(self, num_envs: int = 32, *, max_steps: int = 48, seed: int = 7, visual_period: int = 4):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on sim environment
            raise RuntimeError("MujocoSkillLevelEnv requires the mujoco Python package") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.num_envs, self.max_steps, self.visual_period = num_envs, max_steps, max(1, visual_period)
        self.rng = np.random.default_rng(seed)
        self.step_count = np.zeros(num_envs, dtype=np.int32)
        self.stage = np.zeros(num_envs, dtype=np.int32)  # approach, capture, lift, transport, place
        self.held = np.zeros(num_envs, dtype=bool)
        self.previous_action = np.zeros((num_envs, 8), dtype=np.float32)
        self.previous_recovery = np.zeros(num_envs, dtype=np.int64)
        self.anchor = np.zeros((num_envs, 3), dtype=np.float32)
        self.target = np.zeros((num_envs, 3), dtype=np.float32)
        self.visual_token = np.zeros((num_envs, 6), dtype=np.float32)
        self.pre_lift_z = np.zeros(num_envs, dtype=np.float32)
        self.reset()

    def reset(self, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=np.int64)
        for index in ids:
            data = self.data[int(index)]
            self.mujoco.mj_resetData(self.model, data)
            ball = np.array([self.rng.uniform(.45, .72), self.rng.uniform(-.22, .15), .84], dtype=np.float64)
            data.qpos[:3] = ball
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            curriculum = float(self.rng.random())
            if curriculum < .34:
                hand = [self.rng.uniform(.18, .34), self.rng.uniform(-.18, .18), .96]
                stage = 0
            elif curriculum < .67:
                hand = [ball[0] - .055, ball[1], ball[2] + .015]
                stage = 1
            else:
                hand = [ball[0] - .050, ball[1], ball[2] + .020]
                stage = 2
            data.mocap_pos[0] = hand
            data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
            self.target[index] = [self.rng.uniform(.58, .92), self.rng.uniform(.12, .34), .84]
            self.stage[index] = stage; self.held[index] = stage >= 2; self.step_count[index] = 0; self.previous_action[index] = 0; self.previous_recovery[index] = 0
            self.pre_lift_z[index] = ball[2]
            self.mujoco.mj_forward(self.model, data)
        self._refresh_visual(ids)
        return self.observe()

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        entity = np.zeros((self.num_envs, 2, ENTITY_DIM), dtype=np.float32)
        context = np.zeros((self.num_envs, CONTEXT_DIM), dtype=np.float32)
        for index, data in enumerate(self.data):
            ball = data.xpos[self._body_id("ball")].astype(np.float32)
            hand = data.mocap_pos[0].astype(np.float32)
            entity[index, 0, :3] = ball - hand; entity[index, 0, 3] = .045; entity[index, 0, 4:7] = .09
            entity[index, 0, 7] = float(self.held[index] or self._contact_count(data) > 0)
            entity[index, 1, :3] = self.target[index] - hand; entity[index, 1, 3] = .10
            context[index, :3] = hand; context[index, 3:6] = ball - hand
            context[index, 6] = float(self._contact_count(data)); context[index, 7] = (ball[2] - self.pre_lift_z[index])
            context[index, 8:13] = np.eye(5, dtype=np.float32)[self.stage[index]]
            context[index, 13:21] = self.previous_action[index]
            context[index, 21] = self.previous_recovery[index] / 4.0
            context[index, 22:24] = self.visual_token[index, :2]
        return entity, context

    def step(self, continuous: np.ndarray, recovery: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, SkillInfo]:
        action = np.tanh(np.asarray(continuous, dtype=np.float32)); recovery = np.asarray(recovery, dtype=np.int64)
        rewards = np.zeros(self.num_envs, dtype=np.float32); done = np.zeros(self.num_envs, dtype=bool)
        success = np.zeros(self.num_envs, dtype=bool); dropped = np.zeros(self.num_envs, dtype=bool); contact = np.zeros(self.num_envs, dtype=bool); lifted = np.zeros(self.num_envs, dtype=bool); collision = np.zeros(self.num_envs, dtype=bool)
        for index, data in enumerate(self.data):
            ball_before = data.xpos[self._body_id("ball")].copy()
            hand = data.mocap_pos[0].copy()
            hand_distance_before = np.linalg.norm(ball_before - hand)
            hand += np.array([action[index, 0] * .05 + action[index, 4] * .025, action[index, 1] * .05, action[index, 5] * .018], dtype=np.float64)
            hand[2] = np.clip(hand[2], .79, 1.18)
            close = action[index, 6] > -.15
            if self.stage[index] >= 2 and close:
                hand[2] = min(1.18, hand[2] + .012 + max(0.0, action[index, 7]) * .018)
            data.mocap_pos[0] = hand
            for _ in range(4): self.mujoco.mj_step(self.model, data)
            if self.held[index] and close:
                # Skill-level latch represents a grasp whose contact oracle has
                # already passed. It is not used by the WBC benchmark.
                data.qpos[:3] = hand + np.array([.050, 0.0, -.015])
                data.qvel[:6] = 0.0
                self.mujoco.mj_forward(self.model, data)
            ball = data.xpos[self._body_id("ball")].copy()
            hand_distance_after = np.linalg.norm(ball - hand)
            contacts = self._contact_count(data); contact[index] = contacts > 0
            near = np.linalg.norm(ball[:2] - hand[:2]) < .12
            if self.stage[index] == 0 and near: self.stage[index] = 1
            if self.stage[index] == 1 and contact[index] and close: self.stage[index] = 2; self.held[index] = True; self.pre_lift_z[index] = ball[2]
            dz = ball[2] - self.pre_lift_z[index]; lifted[index] = self.stage[index] >= 2 and dz >= .025 and contact[index]
            if lifted[index]: self.stage[index] = 3
            if self.stage[index] == 3 and np.linalg.norm(ball[:2] - self.target[index, :2]) < .13: self.stage[index] = 4
            success[index] = self.stage[index] == 4 and ball[2] <= .91
            dropped[index] = ball[2] < .62; collision[index] = bool(data.ncon > 8)
            if dropped[index] or (recovery[index] == 4): self.held[index] = False
            progress = np.linalg.norm(ball_before[:2] - self.target[index, :2]) - np.linalg.norm(ball[:2] - self.target[index, :2])
            approach = np.clip(hand_distance_before - hand_distance_after, -.08, .08)
            stage_bonus = .08 * float(self.stage[index] >= 1) + .14 * float(self.stage[index] >= 2) + .20 * float(self.stage[index] >= 3)
            rewards[index] = approach + np.clip(progress, -.2, .2) + .35 * contact[index] + .7 * lifted[index] + stage_bonus + 5.0 * success[index] - 3.0 * dropped[index] - .2 * collision[index] - .04 * (recovery[index] != 0)
            if recovery[index] in (2, 3): self.stage[index] = max(0, self.stage[index] - 1)
            self.step_count[index] += 1; done[index] = success[index] or dropped[index] or self.step_count[index] >= self.max_steps
        self.previous_action = action; self.previous_recovery = recovery
        if np.any(self.step_count % self.visual_period == 0): self._refresh_visual(np.where(self.step_count % self.visual_period == 0)[0])
        finished = np.where(done)[0]
        info = SkillInfo(success, dropped, contact, lifted, collision)
        observation = self.observe()
        if len(finished): self.reset(finished)
        return observation, rewards, done, info

    def _refresh_visual(self, ids: np.ndarray) -> None:
        for index in ids:
            ball = self.data[int(index)].xpos[self._body_id("ball")].astype(np.float32)
            noise = self.rng.normal(0.0, .004, 3).astype(np.float32)
            self.anchor[index] = ball + noise
            self.visual_token[index] = [1.0, 1.0, 1.0, .95, float(self.step_count[index] > 0), ball[2]]

    def _body_id(self, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name))

    def _contact_count(self, data: Any) -> int:
        ball = self._body_id("ball"); hand = self._body_id("hand"); count = 0
        for i in range(int(data.ncon)):
            c = data.contact[i]; a = int(self.model.geom_bodyid[c.geom1]); b = int(self.model.geom_bodyid[c.geom2])
            count += int((a == ball and b == hand) or (b == ball and a == hand))
        return count
