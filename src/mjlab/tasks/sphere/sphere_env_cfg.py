"""Two-sphere payload transport task with acceleration-style effort control."""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_rel,
  joint_vel_l2,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

_SPHERE_XML: Path = Path(__file__).parent / "sphere.xml"
_CTRL_JOINTS = ("tx1", "ty1", "tz1", "tx2", "ty2", "tz2")
_SPHERE_CFG = SceneEntityCfg("sphere", joint_names=_CTRL_JOINTS)
_PAYLOAD_CFG = SceneEntityCfg("sphere", body_names="payload")
_TARGET_POS = (0.6, -0.3, 0.6)
_MASS_EFFECTIVE = 1.25  # sphere mass + half payload mass.
_GRAVITY = 9.81


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_SPHERE_XML))


_SPHERE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlMotorActuatorCfg(target_names_expr=_CTRL_JOINTS),),
)

_SPHERE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "tx1": -0.25,
    "ty1": 0.0,
    "tz1": 0.35,
    "tx2": 0.25,
    "ty2": 0.0,
    "tz2": 0.35,
  },
  joint_vel={".*": 0.0},
)


def _get_sphere_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_SPHERE_ARTICULATION,
    init_state=_SPHERE_INIT,
  )


def position_tracking_reward(
  env,
  asset_cfg: SceneEntityCfg = _PAYLOAD_CFG,
) -> torch.Tensor:
  """Reward payload for staying near a fixed world target position."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
  target_w = torch.tensor(_TARGET_POS, device=env.device, dtype=pos.dtype).unsqueeze(0)
  target_w = target_w + env.scene.env_origins
  pos_error_sq = torch.sum(torch.square(pos - target_w), dim=1)
  return torch.exp(-4.0 * pos_error_sq)


def payload_target_rel_obs(
  env,
  payload_cfg: SceneEntityCfg = _PAYLOAD_CFG,
) -> torch.Tensor:
  targets_w = (
    torch.tensor(_TARGET_POS, device=env.device, dtype=torch.float32).unsqueeze(0)
    + env.scene.env_origins
  )
  asset: Entity = env.scene[payload_cfg.name]
  payload_pos_w = asset.data.body_link_pos_w[:, payload_cfg.body_ids, :].squeeze(1)
  return targets_w - payload_pos_w


def coordinated_action_direction_reward(env) -> torch.Tensor:
  """Encourage both spheres to apply actions in similar directions."""
  actions = env.action_manager.action
  a1 = actions[:, 0:3]
  a2 = actions[:, 3:6]
  cos_sim = torch.sum(a1 * a2, dim=1) / (
    torch.norm(a1, dim=1) * torch.norm(a2, dim=1) + 1e-6
  )
  return 0.5 * (cos_sim + 1.0)


def sphere_translate_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  sphere_cfg = SceneEntityCfg("sphere", joint_names=_CTRL_JOINTS)

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": sphere_cfg},
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": sphere_cfg},
    ),
    "payload_target_rel": ObservationTermCfg(
      func=payload_target_rel_obs,
      params={"payload_cfg": _PAYLOAD_CFG},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "acceleration": JointEffortActionCfg(
      entity_name="sphere",
      actuator_names=_CTRL_JOINTS,
      # Interpret action as desired acceleration delta and convert to force:
      # F = m_eff * a + m_eff * g (z channels only).
      scale=_MASS_EFFECTIVE,
      offset={"tz1": _MASS_EFFECTIVE * _GRAVITY, "tz2": _MASS_EFFECTIVE * _GRAVITY},
    ),
  }

  events = {
    "reset_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-0.05, 0.05),
        "asset_cfg": sphere_cfg,
      },
    ),
  }

  rewards = {
    "target_position": RewardTermCfg(
      func=position_tracking_reward,
      weight=2.0,
      params={"asset_cfg": _PAYLOAD_CFG},
    ),
    "joint_velocity": RewardTermCfg(
      func=joint_vel_l2,
      weight=-0.02,
      params={"asset_cfg": sphere_cfg},
    ),
    "smooth_action": RewardTermCfg(
      func=action_rate_l2,
      weight=-0.05,
    ),
    "action_coordination": RewardTermCfg(
      func=coordinated_action_direction_reward,
      weight=0.1,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={"sphere": _get_sphere_cfg()},
      num_envs=4096,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="sphere",
      body_name="sphere_1",
      distance=2.0,
      elevation=-20.0,
      azimuth=20.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.01, gravity=(0.0, 0.0, -9.81)),
    ),
    decimation=2,
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False

  return cfg


def sphere_translate_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="sphere_translate",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
