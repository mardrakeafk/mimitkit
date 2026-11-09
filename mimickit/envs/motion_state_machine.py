import torch

import util.torch_util as torch_util


def quat_conj(q):
    """xyzw 单位四元数的共轭。"""
    return torch.cat([-q[..., 0:3], q[..., 3:4]], dim=-1)


class MotionStateMachine:
    """管理参考动作的切换。

    切换时记录一个对齐变换（平移 + 绕 z 轴旋转），把新动作的起始帧接到
    「角色当前位置」或「上一帧参考动作所在的位置」，之后每一帧都施加它。
    不做的话，切换瞬间参考 root 会跳回该动作自己的原点。

    对齐目标的优先级：
        1. 传入的 char_root_pos / char_root_rot（训练时用物理角色）
        2. 上一帧的参考 root（可视化时物理角色不动，用这个兜底）
    """

    DEBUG_ALIGN = False         # 确认完改成 False

    def __init__(self, num_envs, motion_lib, device, align_heading=True):
        self._num_envs = num_envs
        self._motion_lib = motion_lib
        self._device = device
        self._align_heading = align_heading

        self._motion_ids = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self._motion_times = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # 记录这一帧发生了切换的 env。
        # 不能靠比较 motion_id —— 随机采样到同一个动作时会漏掉。
        self._just_switched = torch.zeros(num_envs, device=device, dtype=torch.bool)

        # 对齐变换
        self._align_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self._align_quat[:, 3] = 1.0          # 单位四元数 (xyzw)
        self._ref_origin = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self._char_origin = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)

        # 上一帧（变换后）的参考 root。切换时的接续点
        self._last_ref_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self._last_ref_rot = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        self._last_ref_rot[:, 3] = 1.0
        return

    # ---------- 查询 ----------

    def get_motion_ids(self):
        return self._motion_ids

    def get_motion_times(self):
        return self._motion_times

    def is_finished(self):
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        return self._motion_times >= motion_len

    def consume_switched(self):
        """返回这一帧切换了的 env，并清掉标志。"""
        out = self._just_switched.clone()
        self._just_switched[:] = False
        return out

    # ---------- 对齐变换 ----------

    def _clear_align(self, env_ids):
        self._align_quat[env_ids] = 0.0
        self._align_quat[env_ids, 3] = 1.0
        self._ref_origin[env_ids] = 0.0
        self._char_origin[env_ids] = 0.0
        return

    def _set_align(self, env_ids, motion_ids, times,
                   char_root_pos=None, char_root_rot=None):
        ref_pos, ref_rot, _, _, _, _ = self._motion_lib.calc_motion_frame(
            motion_ids, times)

        # 没传角色位姿时（可视化等），接到上一帧的参考动作
        if (char_root_pos is None):
            char_root_pos = self._last_ref_pos[env_ids]
        if (char_root_rot is None):
            char_root_rot = self._last_ref_rot[env_ids]

        if (self._align_heading):
            # 求满足 align * heading(ref) = heading(target) 的旋转
            tgt_h = quat_conj(torch_util.calc_heading_quat_inv(char_root_rot))
            ref_h_inv = torch_util.calc_heading_quat_inv(ref_rot)
            q = torch_util.quat_mul(tgt_h, ref_h_inv)
        else:
            q = torch.zeros((env_ids.shape[0], 4), device=self._device)
            q[:, 3] = 1.0
        self._align_quat[env_ids] = q

        # z 保持参考动作原值，不破坏离地高度
        ref_o = ref_pos.clone()
        ref_o[:, 2] = 0.0
        char_o = char_root_pos.clone()
        char_o[:, 2] = 0.0
        self._ref_origin[env_ids] = ref_o
        self._char_origin[env_ids] = char_o

        if (self.DEBUG_ALIGN):
            d = torch.norm(char_o[:, 0:2] - ref_o[:, 0:2], dim=-1)
            print("[align] 接続先=({:.3f}, {:.3f})  動作原点=({:.3f}, {:.3f})"
                  "  補正={:.3f}m".format(
                      float(char_o[0, 0]), float(char_o[0, 1]),
                      float(ref_o[0, 0]), float(ref_o[0, 1]),
                      float(d.max())), flush=True)
        return

    def transform_frame(self, root_pos, root_rot, root_vel, root_ang_vel):
        """给参考帧施加对齐变换，一次处理全部 env。"""
        d = root_pos - self._ref_origin
        root_pos = torch_util.quat_rotate(self._align_quat, d) + self._char_origin
        root_rot = torch_util.quat_mul(self._align_quat, root_rot)
        root_vel = torch_util.quat_rotate(self._align_quat, root_vel)
        root_ang_vel = torch_util.quat_rotate(self._align_quat, root_ang_vel)

        self._last_ref_pos[:] = root_pos
        self._last_ref_rot[:] = root_rot
        return root_pos, root_rot, root_vel, root_ang_vel

    # ---------- 状态转移 ----------

    def reset(self, env_ids, motion_ids=None, motion_times=None):
        if (motion_ids is not None):
            self._motion_ids[env_ids] = motion_ids
        if (motion_times is not None):
            self._motion_times[env_ids] = motion_times
        else:
            self._motion_times[env_ids] = 0.0

        # reset 时角色会被摆到参考姿势，所以变换取恒等
        self._clear_align(env_ids)

        ref_pos, ref_rot, _, _, _, _ = self._motion_lib.calc_motion_frame(
            self._motion_ids[env_ids], self._motion_times[env_ids])
        self._last_ref_pos[env_ids] = ref_pos
        self._last_ref_rot[env_ids] = ref_rot

        self._just_switched[env_ids] = False
        return

    def switch(self, env_ids, motion_ids, start_time=0.0,
               char_root_pos=None, char_root_rot=None):
        n = env_ids.shape[0]
        if (torch.is_tensor(start_time)):
            times = start_time
        else:
            times = torch.full((n,), float(start_time), device=self._device)

        self._motion_ids[env_ids] = motion_ids
        self._motion_times[env_ids] = times
        self._just_switched[env_ids] = True

        # 无条件对齐。没有角色位姿时 _set_align 内部回退到上一帧的参考
        self._set_align(env_ids, motion_ids, times,
                        char_root_pos, char_root_rot)
        return

    def update(self, dt, char_root_pos=None, char_root_rot=None):
        self._motion_times += dt
        return self._motion_ids, self._motion_times