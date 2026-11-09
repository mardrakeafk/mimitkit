import numpy as np
import os
import sys
import torch

import envs.base_env as base_env
import envs.deepmimic_env as deepmimic_env
import util.torch_util as torch_util


class PushEvalEnv(deepmimic_env.DeepMimicEnv):
    """网格扫描外力的角度和大小,统计抗跌倒成功率。

    每个 env 结束一回合后,现场领取一个还没测够样本的格点继续测,
    直到所有格点都测满 trials 次,输出结果表并退出。
    """

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        self._build_eval_grid(config)
        return

    def _build_eval_grid(self, config):
        cfg = config["env"]["push_eval"]

        angles = torch.arange(cfg["angle_min"], cfg["angle_max"] + 1e-6,
                              cfg["angle_step"], device=self._device,
                              dtype=torch.float32)
        mags = torch.arange(cfg["mag_min"], cfg["mag_max"] + 1e-6,
                            cfg["mag_step"], device=self._device,
                            dtype=torch.float32)

        self._eval_angles = angles
        self._eval_mags = mags
        self._num_angles = angles.shape[0]
        self._num_mags = mags.shape[0]
        self._num_cells = self._num_angles * self._num_mags

        # 格点展平:cell = mag_idx * num_angles + angle_idx
        mm, aa = torch.meshgrid(mags, angles, indexing="ij")
        self._cell_mag = mm.flatten()
        self._cell_angle = torch.deg2rad(aa.flatten())

        self._trials_per_cell = cfg.get("trials", 20)
        self._impact_frame = cfg.get("impact_frame", 16)
        self._impact_jitter = cfg.get("impact_jitter", 3)
        self._push_duration = cfg.get("duration", 2)
        self._init_noise = np.deg2rad(cfg.get("init_noise_deg", 2.0))
        self._out_file = cfg.get("out_file", "output/push_eval.csv")

        self._total_trials = self._num_cells * self._trials_per_cell

        # 统计:已完成 / 已派发未回收
        self._cell_success = torch.zeros(self._num_cells, device=self._device)
        self._cell_total = torch.zeros(self._num_cells, device=self._device)
        self._cell_pending = torch.zeros(self._num_cells, device=self._device)

        num_envs = self.get_num_envs()
        self._env_cell = torch.full((num_envs,), -1, device=self._device,
                                    dtype=torch.long)
        self._eval_done = False
        self._last_report = 0.0

        self._ep_pushed = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._ep_failed = torch.zeros(num_envs, device=self._device, dtype=torch.bool)
        self._ep_impact_frame = torch.full((num_envs,), self._impact_frame,
                                           device=self._device, dtype=torch.long)

        print("[PushEval] 角度 {:.0f}~{:.0f} 步长 {:.0f} → {:d} 个".format(
            cfg["angle_min"], cfg["angle_max"], cfg["angle_step"], self._num_angles))
        print("[PushEval] 力度 {:.0f}~{:.0f} 步长 {:.0f} → {:d} 个".format(
            cfg["mag_min"], cfg["mag_max"], cfg["mag_step"], self._num_mags))
        print("[PushEval] {:d} 格点 × {:d} 次 = {:d} 回合".format(
            self._num_cells, self._trials_per_cell, self._total_trials), flush=True)

        self._assign_cells(torch.arange(num_envs, device=self._device))
        return

    def _assign_cells(self, env_ids):
        """给每个 env 分配一个还差样本的格点;都测满了才标 -1"""
        if (env_ids.shape[0] == 0):
            return

        need = self._trials_per_cell - self._cell_total - self._cell_pending

        for i in range(env_ids.shape[0]):
            e = env_ids[i]
            remaining = torch.where(need > 0)[0]

            if (remaining.shape[0] == 0):
                self._env_cell[e] = -1
            else:
                idx = torch.randint(0, remaining.shape[0], (1,), device=self._device)
                c = remaining[idx]
                self._env_cell[e] = c
                self._cell_pending[c] += 1
                need[c] -= 1
        return

    def _reset_char(self, env_ids):
        # 结算上一回合(构造期首次 reset 时字段还不存在,跳过)
        if (hasattr(self, "_ep_pushed")):
            self._record_results(env_ids)

        super()._reset_char(env_ids)

        if (not hasattr(self, "_ep_pushed")):
            return

        self._ep_pushed[env_ids] = False
        self._ep_failed[env_ids] = False

        # 施力帧抖动,打破确定性
        if (self._impact_jitter > 0):
            jitter = torch.randint(-self._impact_jitter, self._impact_jitter + 1,
                                   (env_ids.shape[0],), device=self._device)
            self._ep_impact_frame[env_ids] = self._impact_frame + jitter

        # 初始姿态噪声
        if (self._init_noise > 0.0):
            char_id = self._get_char_id()
            dof_pos = self._engine.get_dof_pos(char_id)[env_ids]
            noise = torch.randn_like(dof_pos) * self._init_noise
            self._engine.set_dof_pos(env_ids, char_id, dof_pos + noise)
        return

    def _record_results(self, env_ids):
        if (self._eval_done):
            return

        valid = torch.logical_and(self._ep_pushed[env_ids],
                                  self._env_cell[env_ids] >= 0)
        stat_ids = env_ids[valid]

        if (stat_ids.shape[0] > 0):
            cells = self._env_cell[stat_ids]
            ones = torch.ones(cells.shape[0], device=self._device)
            succ = (~self._ep_failed[stat_ids]).float()
            self._cell_total.index_add_(0, cells, ones)
            self._cell_success.index_add_(0, cells, succ)
            self._cell_pending.index_add_(0, cells, -ones)

        # 无效回合把 pending 还回去,那个格点才能被重新派发
        inv_cells = self._env_cell[env_ids[~valid]]
        inv_cells = inv_cells[inv_cells >= 0]
        if (inv_cells.shape[0] > 0):
            self._cell_pending.index_add_(
                0, inv_cells, -torch.ones(inv_cells.shape[0], device=self._device))

        # 所有结束的 env 一律重新领任务(包括之前标 -1 的)
        self._assign_cells(env_ids)

        total = self._cell_total.sum().item()
        if (total - self._last_report >= 50):
            self._last_report = total
            print("[PushEval] {:.0f}/{:d}".format(total, self._total_trials), flush=True)

        if (total >= self._total_trials):
            self._eval_done = True
            self._write_results()
            sys.exit(0)
        return

    def _apply_random_push_from_time(self):
        """覆盖基类:按格点参数施力,每回合只施一次"""
        num_envs = self.get_num_envs()
        char_id = self._get_char_id()

        motion_times = self._get_motion_times()
        fps = self._motion_lib._motion_fps[self._motion_ids]
        curr_frame = (motion_times * fps).long()

        trigger = torch.logical_and(curr_frame >= self._ep_impact_frame,
                                    ~self._ep_pushed)
        trigger = torch.logical_and(trigger, ~self._push_active)
        trigger = torch.logical_and(trigger, self._env_cell >= 0)

        if torch.any(trigger):
            root_rot = self._engine.get_root_rot(char_id)

            local_forward = torch.zeros((num_envs, 3), device=self._device,
                                        dtype=root_rot.dtype)
            local_forward[:, 0] = 1.0
            world_forward = torch_util.quat_rotate(root_rot, local_forward)
            world_forward[:, 2] = 0.0
            world_forward = torch.nn.functional.normalize(world_forward, dim=-1, eps=1e-8)

            base_dir = -world_forward

            cell = self._env_cell.clamp(min=0)
            angle = self._cell_angle[cell]
            magnitude = self._cell_mag[cell]

            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            force_dir = torch.zeros_like(base_dir)
            force_dir[:, 0] = cos_a * base_dir[:, 0] - sin_a * base_dir[:, 1]
            force_dir[:, 1] = sin_a * base_dir[:, 0] + cos_a * base_dir[:, 1]

            push_forces = force_dir * magnitude.unsqueeze(-1)

            self._push_force[trigger] = push_forces[trigger]
            self._push_timer[trigger] = self._push_duration
            self._push_active[trigger] = True
            self._push_body_id[trigger] = self._hit_body_ids[0]
            self._ep_pushed[trigger] = True

        active = self._push_active
        if torch.any(active):
            forces = torch.zeros((num_envs, 3), device=self._device,
                                 dtype=self._push_force.dtype)
            forces[active] = self._push_force[active]

            self._engine.set_body_forces(env_id=None, actor_id=char_id,
                                         body_id=self._push_body_id, forces=forces)

            self._push_timer[active] -= 1
            done = torch.logical_and(self._push_active, self._push_timer <= 0)
            self._push_active[done] = False
            self._push_timer[done] = 0
            self._push_force[done] = 0.0
        return

    def _update_done(self):
        super()._update_done()

        if (not hasattr(self, "_ep_failed")):
            return

        # 动作播完不算结束,让角色有完整时间恢复
        succ = self._done_buf == base_env.DoneFlags.SUCC.value
        self._done_buf[succ] = base_env.DoneFlags.NULL.value

        # 只记施力之后发生的失败
        failed = self._done_buf == base_env.DoneFlags.FAIL.value
        failed = torch.logical_and(failed, self._ep_pushed)
        self._ep_failed = torch.logical_or(self._ep_failed, failed)
        return

    def _write_results(self):
        rate = self._cell_success / self._cell_total.clamp(min=1)
        rate = rate.reshape(self._num_mags, self._num_angles)
        total = self._cell_total.reshape(self._num_mags, self._num_angles)

        rate_np = rate.cpu().numpy()
        total_np = total.cpu().numpy()
        angles_np = self._eval_angles.cpu().numpy()
        mags_np = self._eval_mags.cpu().numpy()

        out_dir = os.path.dirname(self._out_file)
        if (out_dir != ""):
            os.makedirs(out_dir, exist_ok=True)

        with open(self._out_file, "w") as f:
            f.write("mag\\angle," + ",".join("{:.0f}".format(a) for a in angles_np) + "\n")
            for i, m in enumerate(mags_np):
                row = ",".join("{:.3f}".format(v) for v in rate_np[i])
                f.write("{:.0f},{:s}\n".format(m, row))

        print("\n=== 成功率 (行 = 力度 N, 列 = 角度 °) ===")
        print("      " + "".join("{:>7.0f}".format(a) for a in angles_np))
        for i, m in enumerate(mags_np):
            print("{:>5.0f} ".format(m) +
                  "".join("{:>7.2f}".format(v) for v in rate_np[i]))

        print("\n每格样本数 min={:.0f} max={:.0f} (应为 {:d})".format(
            total_np.min(), total_np.max(), self._trials_per_cell))
        print("结果已保存: {:s}".format(self._out_file), flush=True)
        return