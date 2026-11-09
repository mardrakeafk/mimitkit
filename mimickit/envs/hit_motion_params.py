import numpy as np
import os
import torch

import util.torch_util as torch_util


# 动画部位名 → g1 的 body 名
BODY_MAP = {
    "Abdomen":       "waist_roll_link",
    "Chest":         "torso_link",
    "Head":          "head_link",
    "Pelvis":        "pelvis",
    "LeftArm":       "left_elbow_link",
    "RightArm":      "right_elbow_link",
    "LeftShoulder":  "left_shoulder_pitch_link",
    "RightShoulder": "right_shoulder_pitch_link",
    "LeftLeg":       "left_knee_link",
    "RightLeg":      "right_knee_link",
    "LeftFoot":      "left_ankle_roll_link",
    "RightFoot":     "right_ankle_roll_link",
}

# 受击方向 → 角色局部坐标下的受力方向(角色正前方为 +X)
DIR_MAP = {
    "Front": [-1.0,  0.0, 0.0],   # 从正面打来,力指向后方
    "Back":  [ 1.0,  0.0, 0.0],
    "Left":  [ 0.0, -1.0, 0.0],
    "Right": [ 0.0,  1.0, 0.0],
}


def parse_motion_name(filename):
    """AS_HitReaction_Light_Front_LeftArm.pkl → ("light", "Front", "LeftArm")"""
    stem = os.path.splitext(os.path.basename(filename))[0]

    level = None
    direction = None
    body = None

    for p in stem.split("_"):
        if (p in DIR_MAP):
            direction = p
        elif (p in BODY_MAP):
            body = p
        elif (p.lower() in ("light", "medium")):
            level = p.lower()

    if (level is None or direction is None or body is None):
        return None
    return level, direction, body


class HitMotionParams:
    """按 motion 索引保存冲击参数,从动作文件名解析。

    randomize=True  : 力度在 [mag_min, mag_max] 内随机,方向加 ±jitter
    randomize=False : 力度固定为 magnitude,方向不加偏差
    """

    def __init__(self, motion_lib, engine, char_id, config, device):
        self._device = device

        cfg = config["env"]["hit_force"]
        level_cfg = cfg["levels"]
        self._impact_frame = cfg.get("impact_frame", 0)
        self._randomize = cfg.get("randomize", True)
        self._angle_jitter = np.deg2rad(cfg.get("angle_jitter_deg", 0.0))
        self._control_dt = 1.0 / config["engine"]["control_freq"]

        num_motions = motion_lib.get_num_motions()

        self._mag = torch.zeros(num_motions, device=device)
        self._mag_min = torch.zeros(num_motions, device=device)
        self._mag_max = torch.zeros(num_motions, device=device)
        self._frames = torch.zeros(num_motions, device=device, dtype=torch.long)
        self._dir = torch.zeros((num_motions, 3), device=device)
        self._body_id = torch.zeros(num_motions, device=device, dtype=torch.long)
        self._valid = torch.zeros(num_motions, device=device, dtype=torch.bool)

        print("\n[HitForce] 解析 {:d} 个动作 (randomize={})".format(
            num_motions, self._randomize))

        for i in range(num_motions):
            f = motion_lib._motion_files[i]
            parsed = parse_motion_name(f)

            if (parsed is None):
                print("  [{:d}] {:<46s} 解析失败,不施力".format(
                    i, os.path.basename(f)))
                continue

            level, direction, body = parsed
            if (level not in level_cfg):
                print("  [{:d}] {:<46s} 强度 '{:s}' 未配置".format(
                    i, os.path.basename(f), level))
                continue

            g1_body = BODY_MAP[body]
            lv = level_cfg[level]
            n_frames = max(1, int(round(lv["duration"] / self._control_dt)))

            self._mag[i] = lv["magnitude"]
            self._mag_min[i] = lv.get("mag_min", lv["magnitude"])
            self._mag_max[i] = lv.get("mag_max", lv["magnitude"])
            self._frames[i] = n_frames
            self._dir[i] = torch.tensor(DIR_MAP[direction], device=device)
            self._body_id[i] = engine.find_actor_body_id(0, char_id, g1_body)
            self._valid[i] = True

            if (self._randomize):
                rng = "{:.0f}~{:.0f}N".format(self._mag_min[i], self._mag_max[i])
            else:
                rng = "{:.0f}N".format(lv["magnitude"])

            print("  [{:d}] {:<46s} {:<6s} {:<5s} → {:<24s} {:s} × {:.2f}s".format(
                i, os.path.basename(f), level, direction, g1_body,
                rng, lv["duration"]))
        print("", flush=True)
        return

    def get_impact_frame(self):
        return self._impact_frame

    def is_valid(self, motion_ids):
        return self._valid[motion_ids]

    def get_duration_frames(self, motion_ids):
        return self._frames[motion_ids]

    def calc_forces(self, motion_ids, root_rot):
        """返回世界坐标下的力向量和受击 body id"""
        n = motion_ids.shape[0]
        local_dir = self._dir[motion_ids]

        if (self._randomize and self._angle_jitter > 0.0):
            angle = (torch.rand(n, device=self._device) * 2.0 - 1.0) * self._angle_jitter
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            rotated = torch.zeros_like(local_dir)
            rotated[:, 0] = cos_a * local_dir[:, 0] - sin_a * local_dir[:, 1]
            rotated[:, 1] = sin_a * local_dir[:, 0] + cos_a * local_dir[:, 1]
            rotated[:, 2] = local_dir[:, 2]
            local_dir = rotated

        world_dir = torch_util.quat_rotate(root_rot, local_dir)
        world_dir[:, 2] = 0.0
        world_dir = torch.nn.functional.normalize(world_dir, dim=-1, eps=1e-8)

        if (self._randomize):
            lo = self._mag_min[motion_ids]
            hi = self._mag_max[motion_ids]
            mag = torch.rand(n, device=self._device) * (hi - lo) + lo
        else:
            mag = self._mag[motion_ids]

        forces = world_dir * mag.unsqueeze(-1)
        return forces, self._body_id[motion_ids]