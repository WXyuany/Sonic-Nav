import math
import numpy as np
import mujoco


class LidarSim:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str = "lidar",
        num_beams: int = 360,
        max_range: float = 30.0,
        min_range: float = 0.1,
    ):
        self._model = model
        self._data = data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._num_beams = num_beams
        self._max_range = max_range
        self._min_range = min_range
        self._angles = np.linspace(0, 2 * math.pi, num_beams, endpoint=False)
        self._ranges = np.full(num_beams, max_range, dtype=np.float64)
        try:
            self._robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        except Exception:
            self._robot_body = -1

    def step(self):
        pos = self._data.site_xpos[self._site_id].copy()
        rot = self._data.site_xmat[self._site_id].reshape(3, 3)
        forward = rot[:, 0]
        base_angle = math.atan2(forward[1], forward[0])

        for i, angle_offset in enumerate(self._angles):
            world_angle = base_angle + angle_offset
            direction = np.array(
                [math.cos(world_angle), math.sin(world_angle), 0.0], dtype=np.float64
            )
            result = self._ray_cast(pos, direction)
            self._ranges[i] = result if result > 0 else self._max_range

    def _ray_cast(self, origin, direction):
        geom_id = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(
            self._model, self._data,
            origin, direction,
            None, 1, self._robot_body, geom_id,
        )
        if geom_id[0] >= 0:
            geom_pos = self._data.geom_xpos[geom_id[0]]
            dist = float(np.linalg.norm(geom_pos - origin))
            return dist if dist >= self._min_range else self._min_range
        return -1.0

    @property
    def ranges(self):
        return self._ranges.copy()

    @property
    def angles(self):
        return self._angles.copy()

    @property
    def max_range(self):
        return self._max_range

    @property
    def min_range(self):
        return self._min_range


class Mid360Sim:
    def __init__(self, model, data, site_name="lidar", max_range=30.0, min_range=0.1):
        self._model = model
        self._data = data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._max_range = max_range
        self._min_range = min_range
        try:
            self._robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        except Exception:
            self._robot_body = -1
        h_beams, v_beams = 720, 16
        self._h_angles = np.linspace(0, 2*math.pi, h_beams, endpoint=False)
        self._v_angles = np.linspace(-20, 20, v_beams) * math.pi / 180
        self._points = np.zeros((h_beams*v_beams, 3), dtype=np.float32)

    def step(self):
        pos = self._data.site_xpos[self._site_id].copy()
        rot = self._data.site_xmat[self._site_id].reshape(3, 3)
        base_angle = math.atan2(rot[1, 0], rot[0, 0])
        idx = 0
        for va in self._v_angles:
            cv, sv = math.cos(va), math.sin(va)
            for ha in self._h_angles:
                wa = base_angle + ha
                dx = math.cos(wa) * cv
                dy = math.sin(wa) * cv
                dz = sv
                gid = np.array([-1], dtype=np.int32)
                mujoco.mj_ray(self._model, self._data, pos, (dx, dy, dz),
                              None, 1, self._robot_body, gid)
                if gid[0] >= 0:
                    d = np.linalg.norm(self._data.geom_xpos[gid[0]] - pos)
                    d = max(d, self._min_range)
                    self._points[idx] = [pos[0]+dx*d, pos[1]+dy*d, pos[2]+dz*d]
                else:
                    self._points[idx] = [pos[0]+dx*self._max_range, pos[1]+dy*self._max_range, pos[2]+dz*self._max_range]
                idx += 1

    @property
    def points(self): return self._points
    @property
    def max_range(self): return self._max_range
    @property
    def min_range(self): return self._min_range
