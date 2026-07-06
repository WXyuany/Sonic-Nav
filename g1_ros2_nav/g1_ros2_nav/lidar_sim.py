import math
import numpy as np
import mujoco


def _site_body_id(model, site_id):
    return int(model.site_bodyid[site_id])


def _body_id(model, body_name):
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))


def _body_subtree_ids(model, root_body):
    bodies = {int(root_body)}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent in bodies and body_id not in bodies:
                bodies.add(body_id)
                changed = True
    return bodies


def _robot_body_ids(model):
    pelvis_id = _body_id(model, "pelvis")
    if pelvis_id < 0:
        return set()
    return _body_subtree_ids(model, pelvis_id)


def _ray_distance(model, data, origin, direction, body_exclude, skipped_bodies, max_range):
    travelled = 0.0
    ray_origin = origin.copy()
    geom_id = np.array([-1], dtype=np.int32)
    while travelled < max_range:
        geom_id[0] = -1
        distance = mujoco.mj_ray(
            model, data, ray_origin, direction,
            None, 1, body_exclude, geom_id,
        )
        if geom_id[0] < 0 or distance < 0:
            return -1.0
        total = travelled + float(distance)
        if total > max_range:
            return -1.0
        hit_body = int(model.geom_bodyid[int(geom_id[0])])
        if hit_body not in skipped_bodies:
            return total
        step = max(float(distance), 0.0) + 1e-3
        travelled += step
        ray_origin = origin + direction * travelled
    return -1.0


class LidarSim:
    def __init__(self, model, data, site_name="lidar", num_beams=360, max_range=30.0, min_range=0.1):
        self._model = model
        self._data = data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._num_beams = num_beams
        self._max_range = max_range
        self._min_range = min_range
        self._angles = np.linspace(0, 2 * math.pi, num_beams, endpoint=False)
        self._ranges = np.full(num_beams, max_range, dtype=np.float64)
        self._body_exclude = _site_body_id(model, self._site_id)
        self._skipped_bodies = _robot_body_ids(model)

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
        distance = _ray_distance(
            self._model, self._data, origin, direction,
            self._body_exclude, self._skipped_bodies, self._max_range,
        )
        if distance >= 0:
            return max(float(distance), self._min_range)
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
    def __init__(
        self,
        model,
        data,
        site_name="lidar",
        max_range=40.0,
        min_range=0.1,
        horizontal_beams=720,
        channels=28,
    ):
        self._model = model
        self._data = data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"MuJoCo site not found: {site_name}")
        self._max_range = max_range
        self._min_range = min_range
        self._body_exclude = _site_body_id(model, self._site_id)
        self._skipped_bodies = _robot_body_ids(model)
        self._h_angles = np.linspace(0, 2 * math.pi, horizontal_beams, endpoint=False)
        self._v_angles = np.linspace(-7, 52, channels) * math.pi / 180
        self._points = np.zeros((horizontal_beams * channels, 3), dtype=np.float32)
        self._ranges = np.full(horizontal_beams, max_range, dtype=np.float32)
        self._scan_angles = self._h_angles.astype(np.float32)
        self._frame_index = 0

    def step(self):
        pos = self._data.site_xpos[self._site_id].copy()
        rot = self._data.site_xmat[self._site_id].reshape(3, 3)
        idx = 0
        self._ranges.fill(self._max_range)
        phase = (self._frame_index % len(self._h_angles)) * (2 * math.pi / len(self._h_angles)) / 7.0
        self._frame_index += 1
        for line, va in enumerate(self._v_angles):
            cv, sv = math.cos(va), math.sin(va)
            stagger = phase + line * (math.pi / len(self._h_angles))
            for col, ha in enumerate(self._h_angles):
                local_angle = ha + stagger
                local_dir = np.array(
                    [math.cos(local_angle) * cv, math.sin(local_angle) * cv, sv],
                    dtype=np.float64,
                )
                world_dir = rot @ local_dir
                distance = _ray_distance(
                    self._model, self._data, pos, world_dir,
                    self._body_exclude, self._skipped_bodies, self._max_range)
                if distance >= 0:
                    d = min(max(float(distance), self._min_range), self._max_range)
                    self._points[idx] = (local_dir * d).astype(np.float32)
                    horizontal_range = math.hypot(float(self._points[idx, 0]), float(self._points[idx, 1]))
                    if -0.6 <= self._points[idx, 2] <= 0.8:
                        bin_id = int((local_angle % (2 * math.pi)) / (2 * math.pi) * len(self._ranges))
                        bin_id %= len(self._ranges)
                        if horizontal_range < self._ranges[bin_id]:
                            self._ranges[bin_id] = max(horizontal_range, self._min_range)
                else:
                    self._points[idx] = [np.nan, np.nan, np.nan]
                idx += 1

    @property
    def points(self):
        mask = ~np.isnan(self._points[:, 0])
        return self._points[mask]
    @property
    def ranges(self): return self._ranges.copy()
    @property
    def angles(self): return self._scan_angles.copy()
    @property
    def max_range(self): return self._max_range
    @property
    def min_range(self): return self._min_range
