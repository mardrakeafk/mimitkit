import torch

import learning.base_agent as base_agent
import learning.normalizer as normalizer
import learning.ppo_model as ppo_model
import util.torch_util as torch_util
import tkinter as tk
import threading


class SwitchAgent(base_agent.BaseAgent):
    NAME = "SwitchAgent"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        model_config = config["model"]

        # Build two policies and load full bundles (model + per-policy normalizers).
        self.loco_model = ppo_model.PPOModel(model_config, env)
        self.hit_model = ppo_model.PPOModel(model_config, env)
        self.loco_obs_norm, self.loco_a_norm = self._build_policy_normalizers()
        self.hit_obs_norm, self.hit_a_norm = self._build_policy_normalizers()

        self._load_policy_bundle(
            model=self.loco_model,
            obs_norm=self.loco_obs_norm,
            a_norm=self.loco_a_norm,
            model_file=config["loco_model_file"],
        )
        self._load_policy_bundle(
            model=self.hit_model,
            obs_norm=self.hit_obs_norm,
            a_norm=self.hit_a_norm,
            model_file=config["hit_model_file"],
        )

        self.loco_model = self.loco_model.to(self._device)
        self.hit_model = self.hit_model.to(self._device)


        num_envs = self.get_num_envs()
        self._action_dim = self.get_action_size()
        # window to display
        self._current_policy = "None"

        threading.Thread(
            target=self._create_status_window,
            daemon=True
        ).start()

    def _build_optimizer(self, config):
        self._optimizer = None
        return

    def _create_status_window(self):

        self._root = tk.Tk()
        self._root.title("Current Policy")

        self._label = tk.Label(
            self._root,
            text="None",
            font=("Arial", 35)
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
                text=self._current_policy
            )

        self._root.after(
            100,
            self._update_label
        )

    def _build_policy_normalizers(self):
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        obs_norm = normalizer.Normalizer(obs_space.shape, device=self._device, dtype=obs_dtype)

        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)
        if hasattr(a_space, "shape") and len(a_space.shape) > 0:
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, init_std=a_std, dtype=a_dtype)
        else:
            a_mean = torch.tensor([0], device=self._device, dtype=a_dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, init_std=a_std, min_std=0, dtype=a_dtype)

        return obs_norm, a_norm

    def _load_policy_bundle(self, model, obs_norm, a_norm, model_file):
        state_dict = torch.load(model_file, map_location=self._device)

        model_state = {k.replace("_model.", ""): v for k, v in state_dict.items() if k.startswith("_model.")}
        model.load_state_dict(model_state)

        obs_state = {k.replace("_obs_norm.", ""): v for k, v in state_dict.items() if k.startswith("_obs_norm.")}
        if len(obs_state) > 0:
            obs_norm.load_state_dict(obs_state)

        a_state = {k.replace("_a_norm.", ""): v for k, v in state_dict.items() if k.startswith("_a_norm.")}
        if len(a_state) > 0:
            a_norm.load_state_dict(a_state)

    def _decode_policy_action(self, model, obs_norm, a_norm, obs):
        norm_obs = obs_norm.normalize(obs)
        dist = model.eval_actor(norm_obs)
        norm_a = dist.mode
        return a_norm.unnormalize(norm_a)

    def _decide_action(self, obs, info):
        obs = obs.to(self._device)

        num_envs = obs.shape[0]

        action = torch.zeros(
            (num_envs, self._action_dim),
            device=self._device
        )

        # 当前每个env对应的reference motion
        motion_ids = self._env.get_motion_ids()

        # motion0 使用 loco policy
        loco_mask = motion_ids == 0

        # motion1 使用 hit policy
        hit_mask = motion_ids == 1

        if loco_mask.any():
            self._current_policy = "HIT MODEL"
            action[loco_mask] = self._decode_policy_action(
                self.loco_model,
                self.loco_obs_norm,
                self.loco_a_norm,
                obs[loco_mask]
            )

        if hit_mask.any():
            self._current_policy = "flip MODEL"
            action[hit_mask] = self._decode_policy_action(
                self.hit_model,
                self.hit_obs_norm,
                self.hit_a_norm,
                obs[hit_mask]
            )

        return action, {}

    def _record_data_post_step(self, obs, reward, done, info):
        super()._record_data_post_step(obs, reward, done, info)

        done_mask = done != 0
        return
