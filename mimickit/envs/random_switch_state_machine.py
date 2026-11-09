import torch

import envs.motion_state_machine as motion_state_machine


class RandomSwitchStateMachine(motion_state_machine.MotionStateMachine):
    """动作播完后再保持 hold 秒，然后随机切到下一个动作。"""

    def __init__(self, num_envs, motion_lib, device,
                 hold_min=0.5, hold_max=2.0, align_heading=True):
        super().__init__(num_envs=num_envs, motion_lib=motion_lib,
                         device=device, align_heading=align_heading)
        self._hold_min = hold_min
        self._hold_max = hold_max
        self._switch_time = torch.zeros(num_envs, device=device,
                                        dtype=torch.float32)
        all_ids = torch.arange(num_envs, device=device, dtype=torch.int64)
        self._resample_switch_time(all_ids)
        return

    def _resample_switch_time(self, env_ids):
        n = env_ids.shape[0]
        hold = torch.rand(n, device=self._device) \
               * (self._hold_max - self._hold_min) + self._hold_min
        motion_len = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        self._switch_time[env_ids] = motion_len + hold
        return

    def reset(self, env_ids, motion_ids=None, motion_times=None):
        super().reset(env_ids, motion_ids, motion_times)
        self._resample_switch_time(env_ids)
        return

    def update(self, dt, char_root_pos=None, char_root_rot=None):
        self._motion_times += dt

        due = self._motion_times >= self._switch_time
        if torch.any(due):
            env_ids = torch.where(due)[0]
            new_ids = self._motion_lib.sample_motions(env_ids.shape[0])
            pos = None if char_root_pos is None else char_root_pos[env_ids]
            rot = None if char_root_rot is None else char_root_rot[env_ids]
            self.switch(env_ids, new_ids, start_time=0.0,
                        char_root_pos=pos, char_root_rot=rot)
            self._resample_switch_time(env_ids)

        return self._motion_ids, self._motion_times