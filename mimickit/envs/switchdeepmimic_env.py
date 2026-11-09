import os

from isaacgym import gymapi

import envs.deepmimic_env as deepmimic_env

import torch
import tkinter as tk
import threading

class SwitchDeepMimicEnv(deepmimic_env.DeepMimicEnv):

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config,num_envs,device,visualize)

        # motion0 -> motion1 切换时间
        self._switch_time = 0.83

        # 防止重复切换
        self._switched = torch.zeros(
            self._engine.get_num_envs(),
            dtype=torch.bool,
            device=self._device
        )
        self._motion_names = [
            "Push",
            "SpinKick"
        ]

        self._current_motion_name = self._motion_names[0]

        threading.Thread(
            target=self._create_status_window,
            daemon=True
        ).start()

        self._motion_root_offset = torch.zeros(
            (self._engine.get_num_envs(), 3),
            device=self._device
        )

    def get_motion_ids(self):
        return self._motion_ids

    def _create_status_window(self):

        self._root = tk.Tk()
        self._root.title("Current Motion")

        self._label = tk.Label(
            self._root,
            text=self._current_motion_name,
            font=("Arial", 40)
        )

        self._label.pack(
            padx=50,
            pady=50
        )

        self._update_label()

        self._root.mainloop()

    def _update_label(self):

        if hasattr(self, "_label"):
            self._label.config(
                text=self._current_motion_name
            )

        self._root.after(
            100,
            self._update_label
        )

    def _reset_ref_motion(self, env_ids):

        super()._reset_ref_motion(env_ids)
        self._motion_root_offset[env_ids] = 0

        # reset后固定从motion0开始
        self._motion_ids[env_ids] = 0
        self._current_motion_name = self._motion_names[0]
        self._motion_time_offsets[env_ids] = (
            -self._time_buf[env_ids]
        )

        self._switched[env_ids] = False

        return

    def _check_motion_switch(self):

        motion_times = self._get_motion_times()

        switch_mask = (
            (motion_times > self._switch_time)
            & (~self._switched)
        )

        if not torch.any(switch_mask):
            return

        env_ids = torch.nonzero(
            switch_mask,
            as_tuple=False
        ).flatten()

        print("Switch Motion")
        self._current_motion_name = self._motion_names[1]
        old_root_pos = self._ref_root_pos[env_ids].clone()

        self._motion_ids[env_ids] = 1

        self._motion_time_offsets[env_ids] = (
            -self._time_buf[env_ids]
        )

        zero_times = torch.zeros(
            len(env_ids),
            device=self._device
        )

        root_pos, _, _, _, _, _ = \
            self._motion_lib.calc_motion_frame(
                self._motion_ids[env_ids],
                zero_times
            )

        self._motion_root_offset[env_ids] = (
                old_root_pos - root_pos
        )

        self._switched[env_ids] = True

    def _update_ref_motion(self):

        #
        # 先检查是否切换
        #
        self._check_motion_switch()

        #
        # 原版逻辑
        #
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()

        root_pos, root_rot, root_vel, root_ang_vel, \
        joint_rot, dof_vel = \
            self._motion_lib.calc_motion_frame(
                motion_ids,
                motion_times
            )
        root_pos = root_pos + self._motion_root_offset
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        ref_body_pos, _ = \
            self._kin_char_model.forward_kinematics(
                self._ref_root_pos,
                self._ref_root_rot,
                self._ref_joint_rot
            )

        self._ref_body_pos[:] = ref_body_pos

        if self._enable_ref_char():
            dof_pos = self._motion_lib.joint_rot_to_dof(
                joint_rot
            )
            self._ref_dof_pos[:] = dof_pos

        return