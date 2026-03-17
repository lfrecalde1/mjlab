"""Quadrotor payload task with one free quadrotor base and xyz payload slides."""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
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
from mjlab.managers.manager_base import ManagerTermBase
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
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer import ViewerConfig

_XML: Path = Path(__file__).parent / "quadrotor_payload.xml"
_SITE_NAMES = ("thrust", "rateX", "rateY", "rateZ")
_TARGET_POS = (0.0, 0.0, 1.0)
_QUAD_MASS = 0.94
_DQ_EPS = 1e-8


def _quad_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("quadrotor_payload")


def _payload_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("quadrotor_payload", body_names=("payload",))


def _payload_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "quadrotor_payload", joint_names=("payload_x", "payload_y", "payload_z")
  )


def _get_payload_targets(env) -> torch.Tensor:
  target_pos = getattr(env, "_payload_target_pos_w", None)
  if target_pos is None:
    base = torch.tensor(_TARGET_POS, device=env.device, dtype=torch.float32).unsqueeze(0)
    target_pos = base + env.scene.env_origins
    env._payload_target_pos_w = target_pos.clone()
  return target_pos


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


def _dual_h_plus(dq: torch.Tensor) -> torch.Tensor:
  real = dq[:, :4]
  dual = dq[:, 4:8]
  h_real = _h_plus(real)
  h_dual = _h_plus(dual)
  zeros = torch.zeros_like(h_real)
  top = torch.cat([h_real, zeros], dim=2)
  bottom = torch.cat([h_dual, h_real], dim=2)
  return torch.cat([top, bottom], dim=1)


def _dual_quat_conjugate(dq: torch.Tensor) -> torch.Tensor:
  return torch.cat([_quat_conjugate(dq[:, :4]), _quat_conjugate(dq[:, 4:8])], dim=1)


def _dual_quat_pose_error_log(
  current_dq: torch.Tensor, desired_dq: torch.Tensor, eps: float = _DQ_EPS
) -> torch.Tensor:
  desired_conj = _dual_quat_conjugate(desired_dq)
  q_error = torch.bmm(_dual_h_plus(desired_conj), current_dq.unsqueeze(-1)).squeeze(-1)
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
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _quad_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  current_pos_w = asset.data.root_link_pos_w
  current_quat_w = asset.data.root_link_quat_w
  target_w = _get_payload_targets(env).to(dtype=current_pos_w.dtype)
  desired_quat = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=current_quat_w.dtype
  ).unsqueeze(0).expand(env.num_envs, -1)
  current_dq = _pose_to_dual_quat(current_pos_w, current_quat_w)
  desired_dq = _pose_to_dual_quat(target_w, desired_quat)
  return _dual_quat_pose_error_log(current_dq, desired_dq)


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_XML))


_ARTICULATION = EntityArticulationInfoCfg(
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

_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 2.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={"payload_x": 0.0, "payload_y": 0.0, "payload_z": 1.0},
  joint_vel={".*": 0.0},
)


def _get_entity_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=_get_spec, articulation=_ARTICULATION, init_state=_INIT)


def payload_target_rel(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _payload_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  payload_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
  target_w = _get_payload_targets(env).to(dtype=payload_pos_w.dtype)
  return target_w - payload_pos_w


def payload_position_reward(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _payload_cfg()
  pos_error = payload_target_rel(env, asset_cfg=asset_cfg)
  return torch.exp(-2.5 * torch.sum(torch.square(pos_error), dim=1))


def quad_attitude_reward(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  pose_error = dual_quat_pose_error_obs(env, asset_cfg=asset_cfg)
  rot_cost = torch.sum(torch.square(pose_error[:, :3]), dim=1)
  return torch.exp(-4.0 * rot_cost)


def linear_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _quad_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)


def angular_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _quad_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def payload_linear_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _payload_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  payload_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(payload_lin_vel_w), dim=1)


def payload_linear_velocity_obs(
  env,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = _payload_cfg()
  asset: Entity = env.scene[asset_cfg.name]
  payload_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return payload_lin_vel_w


def cable_direction_obs(
  env,
  quad_cfg: SceneEntityCfg | None = None,
  payload_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if quad_cfg is None:
    quad_cfg = _quad_cfg()
  if payload_cfg is None:
    payload_cfg = _payload_cfg()
  asset: Entity = env.scene[quad_cfg.name]
  quad_pos_w = asset.data.root_link_pos_w
  payload_pos_w = asset.data.body_link_pos_w[:, payload_cfg.body_ids, :].squeeze(1)
  cable_vec_w = payload_pos_w - quad_pos_w
  cable_len = torch.linalg.norm(cable_vec_w, dim=1, keepdim=True).clamp_min(1e-6)
  return cable_vec_w / cable_len


def cable_angular_velocity_obs(
  env,
  quad_cfg: SceneEntityCfg | None = None,
  payload_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  if quad_cfg is None:
    quad_cfg = _quad_cfg()
  if payload_cfg is None:
    payload_cfg = _payload_cfg()
  asset: Entity = env.scene[quad_cfg.name]
  quad_pos_w = asset.data.root_link_pos_w
  quad_vel_w = asset.data.root_link_lin_vel_w
  payload_pos_w = asset.data.body_link_pos_w[:, payload_cfg.body_ids, :].squeeze(1)
  payload_vel_w = asset.data.body_link_lin_vel_w[:, payload_cfg.body_ids, :].squeeze(1)

  a = payload_pos_w - quad_pos_w
  a_dot = payload_vel_w - quad_vel_w
  norm_a = torch.linalg.norm(a, dim=1, keepdim=True).clamp_min(1e-6)
  n = a / norm_a
  proj = torch.sum(n * a_dot, dim=1, keepdim=True)
  n_dot = (a_dot - n * proj) / norm_a
  return torch.linalg.cross(n, n_dot, dim=1)


def cable_direction_reward(
  env,
  quad_cfg: SceneEntityCfg | None = None,
  payload_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  n = cable_direction_obs(env, quad_cfg=quad_cfg, payload_cfg=payload_cfg)
  desired = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=n.dtype).unsqueeze(0)
  error_n = torch.linalg.cross(desired.expand_as(n), n, dim=1)
  return torch.exp(-4.0 * torch.sum(torch.square(error_n), dim=1))


def cable_angular_velocity_l2(
  env,
  quad_cfg: SceneEntityCfg | None = None,
  payload_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  r = cable_angular_velocity_obs(env, quad_cfg=quad_cfg, payload_cfg=payload_cfg)
  return torch.sum(torch.square(r), dim=1)


def action_l2(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def reset_payload_below_quadrotor(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg | None = None,
  payload_joint_cfg: SceneEntityCfg | None = None,
) -> None:
  if asset_cfg is None:
    asset_cfg = _quad_cfg()
  if payload_joint_cfg is None:
    payload_joint_cfg = _payload_joint_cfg()
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  quad_pos = env.sim.data.qpos[env_ids[:, None], asset.indexing.free_joint_q_adr[:3]]

  payload_pos = torch.empty((len(env_ids), 3), device=env.device, dtype=quad_pos.dtype)
  xy_offset = torch.empty((len(env_ids), 2), device=env.device, dtype=quad_pos.dtype)
  xy_offset[:, 0].uniform_(-0.2, 0.2)
  xy_offset[:, 1].uniform_(-0.2, 0.2)
  payload_pos[:, 0:2] = quad_pos[:, 0:2] + xy_offset
  payload_pos[:, 2] = quad_pos[:, 2] - 0.8

  payload_vel = torch.zeros_like(payload_pos)
  asset.write_joint_state_to_sim(
    payload_pos,
    payload_vel,
    env_ids=env_ids,
    joint_ids=payload_joint_cfg.joint_ids,
  )


def sample_payload_target(
  env,
  env_ids: torch.Tensor | None,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  target_pos_w = _get_payload_targets(env)
  base_target = torch.tensor(_TARGET_POS, device=env.device, dtype=target_pos_w.dtype)
  target_pos_w[env_ids] = base_target + env.scene.env_origins[env_ids]

  offsets = torch.empty((len(env_ids), 3), device=env.device, dtype=target_pos_w.dtype)
  offsets[:, 0].uniform_(-1.8, 1.8)
  offsets[:, 1].uniform_(-1.8, 1.8)
  offsets[:, 2].uniform_(-1.5, 1.5)
  target_pos_w[env_ids] += offsets


class PayloadTargetVisualizer(ManagerTermBase):
  def __init__(self, cfg: EventTermCfg, env):
    del cfg
    super().__init__(env)

  def __call__(self, env, env_ids, **kwargs) -> None:
    del env, env_ids, kwargs

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    target_pos_w = _get_payload_targets(self._env)
    for idx in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=target_pos_w[idx],
        radius=0.04,
        color=(1.0, 0.2, 0.2, 0.8),
        label=f"payload_target_{idx}",
      )


def quadrotor_payload_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "payload_target_rel": ObservationTermCfg(
      func=payload_target_rel,
      params={"asset_cfg": _payload_cfg()},
    ),
    "payload_lin_vel": ObservationTermCfg(
      func=payload_linear_velocity_obs,
      params={"asset_cfg": _payload_cfg()},
    ),
    "cable_direction": ObservationTermCfg(
      func=cable_direction_obs,
      params={"quad_cfg": _quad_cfg(), "payload_cfg": _payload_cfg()},
    ),
    "cable_ang_vel": ObservationTermCfg(
      func=cable_angular_velocity_obs,
      params={"quad_cfg": _quad_cfg(), "payload_cfg": _payload_cfg()},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel,
      params={"asset_cfg": _quad_cfg()},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel,
      params={"asset_cfg": _quad_cfg()},
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity,
      params={"asset_cfg": _quad_cfg()},
    ),
    "last_action": ObservationTermCfg(
      func=last_action,
      params={"action_name": "body_rate"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms, enable_corruption=True, history_length=4
    ),
    "critic": ObservationGroupCfg({**actor_terms}, history_length=4),
  }

  actions: dict[str, ActionTermCfg] = {
    "body_rate": QuadrotorRateActionCfg(
      entity_name="quadrotor_payload",
      actuator_names=_SITE_NAMES,
      scale={"thrust": 8.0, "rateX": 6.0, "rateY": 6.0, "rateZ": 6.0},
      offset={"thrust": _QUAD_MASS * 9.81},
      clip={
        "thrust": (0.0, 20.0),
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
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.1, 0.1),
          "roll": (-1.5, 1.5),
          "pitch": (-1.5, 1.5),
          "yaw": (-3.14159, 3.14159),
        },
        "velocity_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (-0.1, 0.1),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-0.1, 0.1),
        },
        "asset_cfg": _quad_cfg(),
      },
    ),
    "reset_payload_joints": EventTermCfg(
      func=reset_payload_below_quadrotor,
      mode="reset",
      params={
        "asset_cfg": _quad_cfg(),
        "payload_joint_cfg": _payload_joint_cfg(),
      },
    ),
    "sample_payload_target": EventTermCfg(
      func=sample_payload_target,
      mode="reset",
    ),
    "visualize_payload_target": EventTermCfg(
      func=PayloadTargetVisualizer,
      mode="startup",
    ),
  }

  rewards = {
    "payload_position": RewardTermCfg(
      func=payload_position_reward,
      weight=10.0,
      params={"asset_cfg": _payload_cfg()},
    ),
    "quad_attitude": RewardTermCfg(
      func=quad_attitude_reward,
      weight=3.0,
      params={"asset_cfg": _quad_cfg()},
    ),
    "linear_velocity": RewardTermCfg(
      func=linear_velocity_l2,
      weight=-0.05,
      params={"asset_cfg": _quad_cfg()},
    ),
    "payload_linear_velocity": RewardTermCfg(
      func=payload_linear_velocity_l2,
      weight=-0.05,
      params={"asset_cfg": _payload_cfg()},
    ),
    "cable_direction": RewardTermCfg(
      func=cable_direction_reward,
      weight=0.1,
      params={"quad_cfg": _quad_cfg(), "payload_cfg": _payload_cfg()},
    ),
    "cable_angular_velocity": RewardTermCfg(
      func=cable_angular_velocity_l2,
      weight=-0.05,
      params={"quad_cfg": _quad_cfg(), "payload_cfg": _payload_cfg()},
    ),
    "angular_velocity": RewardTermCfg(
      func=angular_velocity_l2,
      weight=-0.02,
      params={"asset_cfg": _quad_cfg()},
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
      params={"minimum_height": 0.15, "asset_cfg": _quad_cfg()},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"quadrotor_payload": _get_entity_cfg()},
      num_envs=4096,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="quadrotor_payload",
      body_name="drone_0",
      distance=3.0,
      elevation=-20.0,
      azimuth=35.0,
    ),
    sim=SimulationCfg(
      njmax=512,
      nconmax=256,
      mujoco=MujocoCfg(
        timestep=0.01,
        gravity=(0.0, 0.0, -9.81),
        disableflags=("contact",),
      ),
    ),
    decimation=2,
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False

  return cfg


def quadrotor_payload_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
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
    experiment_name="quadrotor_payload",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
