import os
import torch
from isaacgym import gymapi

import envs.motion_state_machine as motion_state_machine


class KeyboardStateMachine(motion_state_machine.MotionStateMachine):
    """数字键切换动作,无输入时停在当前动作最后一帧。"""

    KEYS = [gymapi.KEY_1, gymapi.KEY_2, gymapi.KEY_3,
            gymapi.KEY_4, gymapi.KEY_5, gymapi.KEY_6,
            gymapi.KEY_7, gymapi.KEY_8, gymapi.KEY_9,
            gymapi.KEY_0]

    def __init__(self, num_envs, motion_lib, device, gym, viewer,
                 align_heading=True):
        super().__init__(num_envs, motion_lib, device, align_heading)

        self._gym = gym
        self._viewer = viewer
        self._pending_motion = None

        num_motions = self._motion_lib.get_num_motions()
        self._num_motions = num_motions
        self._num_keys = min(len(self.KEYS), num_motions)

        if (self._viewer is not None):
            for i in range(self._num_keys):
                self._gym.subscribe_viewer_keyboard_event(
                    self._viewer, self.KEYS[i], "motion_{:d}".format(i))

            # 动作超过 10 个时,用 N / B 翻到后面的动作
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_N, "motion_next")
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_B, "motion_prev")

            print("[Keyboard] 已绑定 {:d} 个按键 (共 {:d} 個动作)".format(
                self._num_keys, num_motions))
            for i in range(self._num_keys):
                name = os.path.basename(self._motion_lib._motion_files[i])
                length = self._motion_lib._motion_lengths[i].item()
                key = (i + 1) % 10
                print("  [{:d}] {:s}  ({:.2f}s)".format(key, name, length))
            if (num_motions > self._num_keys):
                print("  [N/B] 下一个 / 上一个动作 (用于第 {:d} 个以后)".format(
                    self._num_keys + 1))
        else:
            print("[Keyboard] viewer 为 None,键盘输入不可用")
        return

    def update(self, dt, char_root_pos=None, char_root_rot=None):
        self._poll_input()

        if (self._pending_motion is not None):
            env_ids = torch.arange(self._num_envs, device=self._device,
                                   dtype=torch.int64)
            motion_ids = torch.full((self._num_envs,), self._pending_motion,
                                    device=self._device, dtype=torch.int64)
            self.switch(env_ids, motion_ids, start_time=0.0,
                        char_root_pos=char_root_pos,
                        char_root_rot=char_root_rot)
            self._pending_motion = None
        else:
            self._motion_times += dt

        return self._motion_ids, self._motion_times

    def _poll_input(self):
        if (self._viewer is None):
            return

        for evt in self._gym.query_viewer_action_events(self._viewer):
            if (evt.value <= 0):
                continue

            if (evt.action == "motion_next"):
                cur = int(self._motion_ids[0])
                self._pending_motion = (cur + 1) % self._num_motions
            elif (evt.action == "motion_prev"):
                cur = int(self._motion_ids[0])
                self._pending_motion = (cur - 1) % self._num_motions
            elif (evt.action.startswith("motion_")):
                self._pending_motion = int(evt.action.split("_")[1])
            else:
                continue

            name = os.path.basename(
                self._motion_lib._motion_files[self._pending_motion])
            print("[key] 切换到动作 {:d}: {:s}".format(
                self._pending_motion, name), flush=True)
        return