import numpy as np
import torch

import envs.base_env as base_env
import envs.view_motion_env as view_motion_env
import envs.keyboard_state_machine as keyboard_state_machine
import envs.random_switch_state_machine as random_switch_state_machine


class ViewStateMachineEnv(view_motion_env.ViewMotionEnv):
    """状態機で参照動作を切り替えながら運動学再生する。

    ViewMotionEnv との違い:
      - 再生時刻と動作 ID を状態機から取る（_time_buf ではない）
      - 切り替え時の整列変換を適用する（原点へ飛ばない）
      - 動作が終わってもエピソードを終了しない
    """

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config=config, num_envs=num_envs,
                         device=device, visualize=visualize)
        self._build_state_machine(config)
        self.reset()
        return

    def _build_state_machine(self, config):
        env_config = config["env"]
        sm_type = env_config.get("state_machine", "keyboard")
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
            self._state_machine = \
                random_switch_state_machine.RandomSwitchStateMachine(
                    num_envs=self.get_num_envs(),
                    motion_lib=self._motion_lib,
                    device=self._device,
                    hold_min=hold_min,
                    hold_max=hold_max,
                    align_heading=align_heading)
            print("[StateMachine] 随机切换 (保持 {:.1f}~{:.1f}s)".format(
                hold_min, hold_max), flush=True)
        return

    def _has_state_machine(self):
        return hasattr(self, "_state_machine")

    def _get_env_motion_ids(self):
        if (not self._has_state_machine()):
            return super()._get_env_motion_ids()
        return self._state_machine.get_motion_ids()

    def _sync_motion(self):
        if (not self._has_state_machine()):
            return super()._sync_motion()

        dt = self._engine.get_timestep() * self._time_scale
        motion_ids, motion_times = self._state_machine.update(dt)

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        # 切り替え時の整列変換（位置・向き）を適用する
        root_pos, root_rot, root_vel, root_ang_vel = \
            self._state_machine.transform_frame(
                root_pos, root_rot, root_vel, root_ang_vel)

        joint_dof = self._motion_lib.joint_rot_to_dof(joint_rot)
        char_id = self._get_char_id()

        self._engine.set_root_pos(None, char_id, root_pos)
        self._engine.set_root_rot(None, char_id, root_rot)
        self._engine.set_root_vel(None, char_id, 0.0)
        self._engine.set_root_ang_vel(None, char_id, 0.0)

        self._engine.set_dof_pos(None, char_id, joint_dof)
        self._engine.set_dof_vel(None, char_id, 0.0)

        if (self._has_key_bodies()):
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot)
            self._ref_body_pos[:] = body_pos
        return

    def _update_done(self):
        if (not self._has_state_machine()):
            return super()._update_done()

        # 動作が終わっても終了しない。CLAMP なら最終フレームを保持し、
        # 次のキー入力まで立位のまま待つ。
        self._done_buf[:] = base_env.DoneFlags.NULL.value
        return