import torch

import envs.base_env as base_env
import envs.deepmimic_env as deepmimic_env
import envs.hit_motion_params as hit_motion_params
import envs.keyboard_state_machine as keyboard_state_machine
import envs.random_switch_state_machine as random_switch_state_machine


class StateMachineEnv(deepmimic_env.DeepMimicEnv):
    """参照动作由状态机驱动的 DeepMimic 环境。

    动作切换的瞬间,按动作文件名对应的部位、方向、力度施加一次冲击。
    训练用随机切换 + 随机力度,查看用键盘切换 + 固定力度;
    两者 obs 结构一致,模型通用。
    """

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        self._build_hit_params(config)
        self._build_state_machine(config)
        self.reset()
        return

    # ---------- 构建 ----------

    def _build_hit_params(self, config):
        self._hit_params = None

        if ("hit_force" not in config["env"]):
            return
        if (not config["env"]["hit_force"].get("enable", True)):
            return

        self._hit_params = hit_motion_params.HitMotionParams(
            motion_lib=self._motion_lib,
            engine=self._engine,
            char_id=self._get_char_id(),
            config=config,
            device=self._device)

        # 这一帧刚发生切换的 env,由 _apply_external_disturbance 消费
        self._just_switched = torch.zeros(self.get_num_envs(),
                                          device=self._device, dtype=torch.bool)
        return

    def _build_state_machine(self, config):
        env_config = config["env"]
        sm_type = env_config.get("state_machine", "random")
        align_heading = env_config.get("align_heading", True)
        viewer = getattr(self._engine, "_viewer", None)

        if (sm_type == "keyboard" and viewer is not None):
            self._state_machine = keyboard_state_machine.KeyboardStateMachine(
                num_envs=self.get_num_envs(),
                motion_lib=self._motion_lib,
                device=self._device,
                gym=self._engine._gym,
                viewer=viewer,
                align_heading=align_heading)
        else:
            hold_min = env_config.get("switch_hold_min", 0.5)
            hold_max = env_config.get("switch_hold_max", 2.0)
            self._state_machine = random_switch_state_machine.RandomSwitchStateMachine(
                num_envs=self.get_num_envs(),
                motion_lib=self._motion_lib,
                device=self._device,
                hold_min=hold_min,
                hold_max=hold_max,
                align_heading=align_heading)
            print("[StateMachine] 随机切换 (保持 {:.1f}~{:.1f}s, 朝向对齐={})".format(
                hold_min, hold_max, align_heading), flush=True)
        return

    def _has_state_machine(self):
        return hasattr(self, "_state_machine")

    def _has_hit_params(self):
        return getattr(self, "_hit_params", None) is not None

    # ---------- 参照动作 ----------

    def _get_motion_times(self, env_ids=None):
        if (not self._has_state_machine()):
            return super()._get_motion_times(env_ids)

        times = self._state_machine.get_motion_times()
        if (env_ids is not None):
            times = times[env_ids]
        return times

    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        # 新 episode 开始播新动作,视作一次切换
        if (self._has_hit_params()):
            self._just_switched[env_ids] = False
        return

    def _reset_ref_motion(self, env_ids):
        if (not self._has_state_machine()):
            return super()._reset_ref_motion(env_ids)

        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)

        self._state_machine.reset(env_ids, motion_ids, motion_times)
        self._motion_ids[env_ids] = motion_ids

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel

        ref_body_pos, _ = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos

        self._ref_dof_pos[env_ids] = self._motion_lib.joint_rot_to_dof(joint_rot)
        return

    def _update_ref_motion(self):
        if (not self._has_state_machine()):
            return super()._update_ref_motion()

        dt = self._engine.get_timestep()
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_root_rot = self._engine.get_root_rot(char_id)

        motion_ids, motion_times = self._state_machine.update(
            dt, char_root_pos, char_root_rot)

        # 切换由状态机自己报告，不再比较 motion_id
        if (self._has_hit_params()):
            self._just_switched = torch.logical_or(
                self._just_switched, self._state_machine.consume_switched())

        self._motion_ids[:] = motion_ids

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        root_pos, root_rot, root_vel, root_ang_vel = \
            self._state_machine.transform_frame(
                root_pos, root_rot, root_vel, root_ang_vel)

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        ref_body_pos, _ = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos

        if (self._enable_ref_char()):
            self._ref_dof_pos[:] = self._motion_lib.joint_rot_to_dof(joint_rot)
        return

    # ---------- 外力 ----------

    def _apply_external_disturbance(self):
        if (not self._has_hit_params()):
            return

        num_envs = self.get_num_envs()
        char_id = self._get_char_id()

        # 只在动作刚切换的那一帧施力
        trigger = torch.logical_and(
            self._just_switched, self._hit_params.is_valid(self._motion_ids))
        self._just_switched[:] = False

        if torch.any(trigger):
            root_rot = self._engine.get_root_rot(char_id)
            forces, body_ids = self._hit_params.calc_forces(
                self._motion_ids, root_rot)
            frames = self._hit_params.get_duration_frames(self._motion_ids)

            self._push_force[trigger] = forces[trigger]
            self._push_body_id[trigger] = body_ids[trigger]
            self._push_timer[trigger] = frames[trigger].int()
            self._push_active[trigger] = True

        active = self._push_active
        if torch.any(active):
            forces = torch.zeros((num_envs, 3), device=self._device,
                                 dtype=self._push_force.dtype)
            forces[active] = self._push_force[active]

            self._engine.set_body_forces(env_id=None, actor_id=char_id,
                                         body_id=self._push_body_id,
                                         forces=forces)

            self._push_timer[active] -= 1
            done = torch.logical_and(self._push_active, self._push_timer <= 0)
            self._push_active[done] = False
            self._push_timer[done] = 0
            self._push_force[done] = 0.0
        return

    # ---------- 终止 ----------

    def _update_done(self):
        super()._update_done()

        if (not self._has_state_machine()):
            return

        # 动作播完由状态机接管切换,不结束 episode
        succ = self._done_buf == base_env.DoneFlags.SUCC.value
        self._done_buf[succ] = base_env.DoneFlags.NULL.value
        return