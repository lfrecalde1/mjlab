"""Quadrotor position-tracking task with an inner body-rate controller."""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  last_action,
  projected_gravity,
  reset_root_state_uniform,
  root_height_below_minimum,
  time_out,
)
from mjlab.envs.mdp.actions import QuadrotorRateActionCfg
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_QUAD_XML: Path = Path(__file__).parent / "quadrotor.xml"
_QUAD_CFG = SceneEntityCfg("quadrotor")
_QUAD_MASS = 0.94
_TARGET_POS = (0.0, 0.0, 1.0)
_SITE_NAMES = ("thrust", "rateX", "rateY", "rateZ")
_DQ_EPS = 1e-8


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_QUAD_XML))


_QUAD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMotorActuatorCfg(
      target_names_expr=("thrust",), transmission_type=TransmissionType.SITE
    ),
    XmlMotorActuatorCfg(
      target_names_expr=("rateX",), transmission_type=TransmissionType.SITE
    ),
    XmlMotorActuatorCfg(
      target_names_expr=("rateY",), transmission_type=TransmissionType.SITE
    ),
    XmlMotorActuatorCfg(
      target_names_expr=("rateZ",), transmission_type=TransmissionType.SITE
    ),
  ),
)

_QUAD_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8),
  rot=(1.0, 0.0, 0.0, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
)


def _get_quadrotor_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_QUAD_ARTICULATION,
    init_state=_QUAD_INIT,
  )


def target_position_rel(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target_w = (
    torch.tensor(_TARGET_POS, device=env.device, dtype=torch.float32).unsqueeze(0)
    + env.scene.env_origins
  )
  return target_w - asset.data.root_link_pos_w


def _h_plus(q: torch.Tensor) -> torch.Tensor:
  w = q[:, 0]
  x = q[:, 1]
  y = q[:, 2]
  z = q[:, 3]
  row0 = torch.stack([w, -x, -y, -z], dim=1)
  row1 = torch.stack([x, w, -z, y], dim=1)
  row2 = torch.stack([y, z, w, -x], dim=1)
  row3 = torch.stack([z, -y, x, w], dim=1)
  return torch.stack([row0, row1, row2, row3], dim=1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[:, :1], -q[:, 1:4]], dim=1)


def _quat_left_product(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
  return torch.bmm(_h_plus(q), p.unsqueeze(-1)).squeeze(-1)


def _pose_to_dual_quat(pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
  trans_quat = torch.cat([torch.zeros_like(pos_w[:, :1]), pos_w], dim=1)
  dual = 0.5 * _quat_left_product(trans_quat, quat_w)
  return torch.cat([quat_w, dual], dim=1)


def _dual_quat_conjugate(qd: torch.Tensor) -> torch.Tensor:
  return torch.cat(
    [_quat_conjugate(qd[:, :4]), _quat_conjugate(qd[:, 4:8])], dim=1
  )


def _dual_h_plus(qd: torch.Tensor) -> torch.Tensor:
  real = qd[:, :4]
  dual = qd[:, 4:8]
  h_real = _h_plus(real)
  h_dual = _h_plus(dual)
  zeros = torch.zeros_like(h_real)
  top = torch.cat([h_real, zeros], dim=2)
  bottom = torch.cat([h_dual, h_real], dim=2)
  return torch.cat([top, bottom], dim=1)


def _dual_quat_pose_error_log(
  current_dq: torch.Tensor, desired_dq: torch.Tensor, eps: float = _DQ_EPS
) -> torch.Tensor:
  desired_conj = _dual_quat_conjugate(desired_dq)
  q_error = torch.bmm(
    _dual_h_plus(desired_conj), current_dq.unsqueeze(-1)
  ).squeeze(-1)
  q_error_real = q_error[:, :4]
  q_error_dual = q_error[:, 4:8]
  q_error_real_conj = _quat_conjugate(q_error_real)

  imag = q_error_real[:, 1:4]
  imag_norm = torch.linalg.norm(imag + eps, dim=1, keepdim=True)
  angle = torch.atan2(imag_norm, q_error_real[:, 0:1])

  trans_error = 2.0 * _quat_left_product(q_error_dual, q_error_real_conj)
  log_quat = 0.5 * angle * imag / imag_norm
  log_trans = 0.5 * trans_error[:, 1:4]
  return torch.cat([log_quat, log_trans], dim=1)


def dual_quat_pose_error_obs(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  current_pos_w = asset.data.root_link_pos_w
  current_quat_w = asset.data.root_link_quat_w
  target_w = (
    torch.tensor(_TARGET_POS, device=env.device, dtype=current_pos_w.dtype).unsqueeze(0)
    + env.scene.env_origins
  )
  desired_quat = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=current_quat_w.dtype
  ).unsqueeze(0).expand(env.num_envs, -1)
  current_dq = _pose_to_dual_quat(current_pos_w, current_quat_w)
  desired_dq = _pose_to_dual_quat(target_w, desired_quat)
  return _dual_quat_pose_error_log(current_dq, desired_dq)


def dual_quat_pose_reward(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  pose_error = dual_quat_pose_error_obs(env, asset_cfg=asset_cfg)
  rot_cost = torch.sum(torch.square(pose_error[:, :3]), dim=1)
  trans_cost = torch.sum(torch.square(pose_error[:, 3:6]), dim=1)
  return torch.exp(-(4.0 * rot_cost + 2.5 * trans_cost))


def linear_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)


def angular_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def action_l2(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def _make_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "dq_pose_error": ObservationTermCfg(
      func=dual_quat_pose_error_obs,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "last_action": ObservationTermCfg(
      func=last_action,
      params={"action_name": "body_rate"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "body_rate": QuadrotorRateActionCfg(
      entity_name="quadrotor",
      actuator_names=_SITE_NAMES,
      scale={"thrust": 8.0, "rateX": 6.0, "rateY": 6.0, "rateZ": 6.0},
      offset={"thrust": _QUAD_MASS * 9.81},
      clip={
        "thrust": (0.0, 60.0),
        "rateX": (-6.0, 6.0),
        "rateY": (-6.0, 6.0),
        "rateZ": (-6.0, 6.0),
      },
      preserve_order=True,
    ),
  }

  events = {
    "reset_root": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.2, 0.2),
          "roll": (-2.0, 2.0),
          "pitch": (-2.0, 2.0),
          "yaw": (-3.14159, 3.14159),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.2, 0.2),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
        "asset_cfg": _QUAD_CFG,
      },
    ),
  }

  rewards = {
    "target_pose": RewardTermCfg(
      func=dual_quat_pose_reward,
      weight=4.0,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "linear_velocity": RewardTermCfg(
      func=linear_velocity_l2,
      weight=-0.05,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "angular_velocity": RewardTermCfg(
      func=angular_velocity_l2,
      weight=-0.02,
      params={"asset_cfg": _QUAD_CFG},
    ),
    "smooth_action": RewardTermCfg(
      func=action_rate_l2,
      weight=-0.05,
    ),
    "action_magnitude": RewardTermCfg(
      func=action_l2,
      weight=-0.01,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "too_low": TerminationTermCfg(
      func=root_height_below_minimum,
      params={"minimum_height": 0.15, "asset_cfg": _QUAD_CFG},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"quadrotor": _get_quadrotor_cfg()},
      num_envs=4096,
      env_spacing=2.5,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="quadrotor",
      body_name="drone_0",
      distance=2.0,
      elevation=-25.0,
      azimuth=35.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.01,
        gravity=(0.0, 0.0, -9.81),
      )
    ),
    decimation=2,
    episode_length_s=10.0,
  )


def quadrotor_hover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg()
  if play:
    cfg.episode_length_s = 1e10
    cfg.scene.num_envs = 64
    cfg.observations["actor"].enable_corruption = False
  return cfg


def quadrotor_hover_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128),
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
    experiment_name="quadrotor_hover",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
