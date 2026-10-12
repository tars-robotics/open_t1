# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vec-env wrapper for Isaac Lab environments targeting the vanilla ``rsl_rl`` (rsl_rl_lib >= 5.5.0) API.

``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` (IsaacLab 2.2.0) implements the OLD rsl-rl-lib (~2.3.3) contract:
``get_observations()``/``reset()``/``step()`` return a bare ``"policy"`` tensor plus an ``extras["observations"]``
dict with the other groups. The vanilla ``rsl_rl.env.VecEnv`` abstract base (used by
``rsl_rl.runners.OnPolicyRunner``/``rsl_rl.algorithms.PPO.construct_algorithm``/``rsl_rl.utils.resolve_obs_groups``)
instead expects a single ``TensorDict`` containing ALL observation groups (e.g. "policy", "critic") on every call,
so that ``obs_groups`` (actor/critic -> list of group names) can select from it directly.

This wrapper is otherwise a straight port of ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper``.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps an Isaac Lab environment to be compatible with the vanilla ``rsl_rl`` API."""

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        self.env = env
        self.clip_actions = clip_actions

        # store information required by the vanilla rsl_rl VecEnv interface
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        self.cfg = self.unwrapped.cfg

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def _obs_dict_to_tensordict(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        """Wrap a plain observation-group dict as a ``TensorDict`` for the vanilla ``rsl_rl`` API."""
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment, keyed by observation group."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._obs_dict_to_tensordict(obs_dict)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> TensorDict:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return self._obs_dict_to_tensordict(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = self._obs_dict_to_tensordict(obs_dict)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
