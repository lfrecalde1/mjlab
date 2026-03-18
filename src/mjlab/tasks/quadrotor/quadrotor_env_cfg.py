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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from mjlab.sensor import CameraSensor, CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer import ViewerConfig

_QUAD_XML: Path = Path(__file__).parent / "quadrotor.xml"
_QUAD_CFG = SceneEntityCfg("quadrotor")
_QUAD_MASS = 0.94
_TARGET_POS = (0.0, 0.0, 1.0)
_SITE_NAMES = ("thrust", "rateX", "rateY", "rateZ")
_DQ_EPS = 1e-8
_OBSTACLE_SPHERE_SPECS = (
  ("sphere_0", 0.10, (0.2, -0.6, 0.95), (1.0, 0.2, 0.2, 0.9)),
  ("sphere_1", 0.11, (0.7, 0.0, 1.0), (0.2, 0.9, 0.2, 0.9)),
  ("sphere_2", 0.09, (1.1, 0.7, 1.05), (0.2, 0.4, 1.0, 0.9)),
  ("sphere_3", 0.12, (-0.2, 0.5, 0.92), (0.95, 0.75, 0.2, 0.9)),
  ("sphere_4", 0.10, (0.4, 0.9, 1.08), (0.75, 0.2, 0.95, 0.9)),
  ("sphere_5", 0.11, (-0.6, -0.3, 0.98), (0.2, 0.9, 0.9, 0.9)),
  ("sphere_6", 0.09, (0.9, -0.9, 1.02), (1.0, 0.55, 0.15, 0.9)),
  ("sphere_7", 0.10, (-0.8, 0.2, 0.96), (0.45, 1.0, 0.35, 0.9)),
  ("sphere_8", 0.11, (0.0, 1.0, 1.04), (0.3, 0.8, 1.0, 0.9)),
  ("sphere_9", 0.09, (-1.0, -0.8, 0.94), (0.95, 0.3, 0.75, 0.9)),
)
_SPHERE_ENTITY_NAMES = tuple(name for name, _, _, _ in _OBSTACLE_SPHERE_SPECS)
_SPHERE_RADII = tuple(radius for _, radius, _, _ in _OBSTACLE_SPHERE_SPECS)


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


def _get_depth_obstacle_cfg() -> EntityCfg:
  def _build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "quadrotor_depth_obstacles"

    wall_rgba = (0.72, 0.74, 0.78, 1.0)
    box_rgba = (0.82, 0.46, 0.24, 1.0)
    accent_rgba = (0.28, 0.62, 0.84, 1.0)

    geoms = [
      ("front_wall", mujoco.mjtGeom.mjGEOM_BOX, (2.4, 0.12, 1.2), (2.5, 0.0, 1.2), wall_rgba),
      ("left_wall", mujoco.mjtGeom.mjGEOM_BOX, (0.12, 2.2, 1.2), (0.0, 2.1, 1.2), wall_rgba),
      ("right_wall", mujoco.mjtGeom.mjGEOM_BOX, (0.12, 2.2, 1.2), (0.0, -2.1, 1.2), wall_rgba),
      ("pillar_a", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.18, 1.4, 0.0), (1.3, 0.75, 1.4), accent_rgba),
      ("pillar_b", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.16, 1.1, 0.0), (1.8, -0.95, 1.1), accent_rgba),
      ("box_a", mujoco.mjtGeom.mjGEOM_BOX, (0.35, 0.35, 0.35), (1.1, 0.0, 0.35), box_rgba),
      ("box_b", mujoco.mjtGeom.mjGEOM_BOX, (0.28, 0.45, 0.6), (1.7, 1.1, 0.6), box_rgba),
      ("box_c", mujoco.mjtGeom.mjGEOM_BOX, (0.42, 0.3, 0.45), (2.0, -1.15, 0.45), box_rgba),
      ("table", mujoco.mjtGeom.mjGEOM_BOX, (0.65, 0.45, 0.08), (1.55, 0.0, 0.82), (0.5, 0.38, 0.24, 1.0)),
    ]

    for name, geom_type, size, pos, rgba in geoms:
      geom = spec.worldbody.add_geom()
      geom.name = name
      geom.type = geom_type
      geom.size[:] = size
      geom.pos[:] = pos
      geom.rgba[:] = rgba

    return spec

  return EntityCfg(spec_fn=_build_spec)


def _get_obstacle_sphere_cfg(
  name: str,
  radius: float,
  default_pos: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
) -> EntityCfg:
  def _build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = name

    body = spec.worldbody.add_body(name="sphere_body")
    geom = body.add_geom()
    geom.name = "sphere_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size[:] = (radius, 0.0, 0.0)
    geom.rgba[:] = rgba
    geom.contype = 0
    geom.conaffinity = 0
    return spec

  return EntityCfg(
    spec_fn=_build_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
  )


def _get_quadrotor_targets(env) -> torch.Tensor:
  target_pos = getattr(env, "_quadrotor_target_pos_w", None)
  if target_pos is None:
    base = torch.tensor(_TARGET_POS, device=env.device, dtype=torch.float32).unsqueeze(0)
    target_pos = base + env.scene.env_origins
    env._quadrotor_target_pos_w = target_pos.clone()
  return target_pos


def target_position_rel(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target_w = _get_quadrotor_targets(env)
  return target_w - asset.data.root_link_pos_w


def desired_world_velocity(
  env,
  desired_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
  return torch.tensor(
    desired_velocity, device=env.device, dtype=torch.float32
  ).unsqueeze(0).expand(env.num_envs, -1)


def root_linear_velocity_world(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


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
  target_w = _get_quadrotor_targets(env).to(dtype=current_pos_w.dtype)
  desired_quat = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=current_quat_w.dtype
  ).unsqueeze(0).expand(env.num_envs, -1)
  current_dq = _pose_to_dual_quat(current_pos_w, current_quat_w)
  desired_dq = _pose_to_dual_quat(target_w, desired_quat)
  return _dual_quat_pose_error_log(current_dq, desired_dq)


def sample_quadrotor_target(
  env,
  env_ids: torch.Tensor | None,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  target_pos_w = _get_quadrotor_targets(env)
  offsets = torch.empty((len(env_ids), 3), device=env.device, dtype=target_pos_w.dtype)
  offsets[:, 0].uniform_(-1.0, 1.0)
  offsets[:, 1].uniform_(-1.0, 1.0)
  offsets[:, 2].uniform_(1.0, 2.0)
  target_pos_w[env_ids] = env.scene.env_origins[env_ids] + offsets


def sample_obstacle_sphere_near_target(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  xy_min_radius: float = 0.8,
  xy_max_radius: float = 1.4,
  z_half_range: float = 0.12,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  sphere: Entity = env.scene[asset_cfg.name]
  target_pos_w = _get_quadrotor_targets(env)[env_ids]
  pose = torch.zeros((len(env_ids), 7), device=env.device, dtype=target_pos_w.dtype)
  pose[:, 0:3] = target_pos_w
  theta = torch.empty(len(env_ids), device=env.device, dtype=target_pos_w.dtype)
  theta.uniform_(0.0, 2.0 * torch.pi)
  radius = torch.empty(len(env_ids), device=env.device, dtype=target_pos_w.dtype)
  radius.uniform_(xy_min_radius, xy_max_radius)
  pose[:, 0].add_(radius * torch.cos(theta))
  pose[:, 1].add_(radius * torch.sin(theta))
  pose[:, 2].add_(torch.empty(len(env_ids), device=env.device).uniform_(-z_half_range, z_half_range))
  pose[:, 3] = 1.0
  sphere.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def obstacle_spawn_curriculum(
  env,
  env_ids: torch.Tensor,
  sphere_names: tuple[str, ...] = _SPHERE_ENTITY_NAMES,
  stages: list[dict[str, float]] | None = None,
) -> dict[str, torch.Tensor]:
  del env_ids
  if stages is None:
    stages = [
      {"step": 0, "xy_min_radius": 3.2, "xy_max_radius": 3.9},
      {"step": 5_000, "xy_min_radius": 0.9, "xy_max_radius": 1.6},
      {"step": 15_000, "xy_min_radius": 0.6, "xy_max_radius": 1.3},
      {"step": 30_000, "xy_min_radius": 0.3, "xy_max_radius": 1.1},
    ]

  selected = stages[0]
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      selected = stage

  for name in sphere_names:
    term_cfg = env.event_manager.get_term_cfg(f"reset_{name}")
    term_cfg.params["xy_min_radius"] = selected["xy_min_radius"]
    term_cfg.params["xy_max_radius"] = selected["xy_max_radius"]

  return {
    "xy_min_radius": torch.tensor(selected["xy_min_radius"]),
    "xy_max_radius": torch.tensor(selected["xy_max_radius"]),
  }


def sample_obstacle_sphere_on_velocity_path(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  x_range: tuple[float, float] = (2.5, 4.5),
  y_half_range: float = 1.4,
  z_range: tuple[float, float] = (1.0, 2.0),
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  sphere: Entity = env.scene[asset_cfg.name]
  pose = torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32)
  pose[:, 0:3] = env.scene.env_origins[env_ids]
  pose[:, 0].add_(torch.empty(len(env_ids), device=env.device).uniform_(*x_range))
  pose[:, 1].add_(torch.empty(len(env_ids), device=env.device).uniform_(-y_half_range, y_half_range))
  pose[:, 2].add_(torch.empty(len(env_ids), device=env.device).uniform_(*z_range))
  pose[:, 3] = 1.0
  sphere.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def velocity_obstacle_curriculum(
  env,
  env_ids: torch.Tensor,
  sphere_names: tuple[str, ...] = _SPHERE_ENTITY_NAMES,
  stages: list[dict[str, float]] | None = None,
) -> dict[str, torch.Tensor]:
  del env_ids
  if stages is None:
    stages = [
      {"step": 0, "x_min": 2.5, "x_max": 4.5},
      {"step": 5_000, "x_min": 1.8, "x_max": 4.0},
      {"step": 15_000, "x_min": 1.0, "x_max": 3.4},
      {"step": 30_000, "x_min": 0.4, "x_max": 2.8},
    ]

  selected = stages[0]
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      selected = stage

  for name in sphere_names:
    term_cfg = env.event_manager.get_term_cfg(f"reset_{name}")
    term_cfg.params["x_range"] = (selected["x_min"], selected["x_max"])

  return {
    "x_min": torch.tensor(selected["x_min"]),
    "x_max": torch.tensor(selected["x_max"]),
  }


class QuadrotorTargetVisualizer(ManagerTermBase):
  def __init__(self, cfg: EventTermCfg, env):
    del cfg
    super().__init__(env)

  def __call__(self, env, env_ids, **kwargs) -> None:
    del env, env_ids, kwargs

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    target_pos_w = _get_quadrotor_targets(self._env)
    for idx in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=target_pos_w[idx],
        radius=0.05,
        color=(1.0, 0.1, 0.1, 0.9),
        label=f"quadrotor_target_{idx}",
      )


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


def world_velocity_tracking_reward(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
  desired_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
  std: float = 0.5,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  desired = torch.tensor(
    desired_velocity,
    device=env.device,
    dtype=asset.data.root_link_lin_vel_w.dtype,
  ).unsqueeze(0)
  vel_error = asset.data.root_link_lin_vel_w - desired
  return torch.exp(-torch.sum(torch.square(vel_error), dim=1) / std**2)


def angular_velocity_l2(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def action_l2(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def camera_depth_obs(
  env,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


def sphere_positions_body_frame(
  env,
  sphere_names: tuple[str, ...] = _SPHERE_ENTITY_NAMES,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quad_pos_w = asset.data.root_link_pos_w
  quad_quat_w = asset.data.root_link_quat_w

  rel_positions_b: list[torch.Tensor] = []
  for sphere_name in sphere_names:
    sphere: Entity = env.scene[sphere_name]
    rel_pos_w = sphere.data.root_link_pos_w - quad_pos_w
    rel_positions_b.append(quat_apply_inverse(quad_quat_w, rel_pos_w))

  return torch.cat(rel_positions_b, dim=1)


def sphere_radii_obs(
  env,
  radii: tuple[float, ...] = _SPHERE_RADII,
) -> torch.Tensor:
  return torch.tensor(radii, device=env.device, dtype=torch.float32).unsqueeze(0).expand(
    env.num_envs, -1
  )


def obstacle_avoidance_penalty(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
  sphere_names: tuple[str, ...] = _SPHERE_ENTITY_NAMES,
  sphere_radii: tuple[float, ...] = _SPHERE_RADII,
  safety_margin: float = 0.55,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quad_pos_w = asset.data.root_link_pos_w

  clearances: list[torch.Tensor] = []
  for sphere_name, radius in zip(sphere_names, sphere_radii, strict=True):
    sphere: Entity = env.scene[sphere_name]
    center_distance = torch.linalg.norm(sphere.data.root_link_pos_w - quad_pos_w, dim=1)
    clearances.append(center_distance - radius)

  clearance_tensor = torch.stack(clearances, dim=1)
  violations = torch.clamp(safety_margin - clearance_tensor, min=0.0)
  return torch.sum(torch.square(violations), dim=1)


def sphere_overlap_termination(
  env,
  asset_cfg: SceneEntityCfg = _QUAD_CFG,
  sphere_names: tuple[str, ...] = _SPHERE_ENTITY_NAMES,
  sphere_radii: tuple[float, ...] = _SPHERE_RADII,
  quadrotor_radius: float = 0.14,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quad_pos_w = asset.data.root_link_pos_w

  min_clearance: torch.Tensor | None = None
  for sphere_name, radius in zip(sphere_names, sphere_radii, strict=True):
    sphere: Entity = env.scene[sphere_name]
    center_distance = torch.linalg.norm(sphere.data.root_link_pos_w - quad_pos_w, dim=1)
    clearance = center_distance - (radius + quadrotor_radius)
    min_clearance = (
      clearance
      if min_clearance is None
      else torch.minimum(min_clearance, clearance)
    )

  assert min_clearance is not None
  return min_clearance <= 0.0


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
          "roll": (-1.5, 1.5),
          "pitch": (-1.5, 1.5),
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
      num_envs=1000,
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


def quadrotor_hover_depth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = quadrotor_hover_env_cfg(play=play)
  cfg.scene.entities["obstacles"] = _get_depth_obstacle_cfg()
  cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "contact")

  depth_cam = CameraSensorCfg(
    name="depth_cam",
    camera_name="quadrotor/drone_0_camera",
    fovy=80.0,
    width=200,
    height=100,
    data_types=("rgb","depth",),
    use_shadows=False,
    use_textures=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_cam,)

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "depth_cam": ObservationTermCfg(
        func=camera_depth_obs,
        params={"sensor_name": depth_cam.name, "cutoff_distance": 10.0},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  if play:
    cfg.scene.num_envs = 16

  return cfg


def quadrotor_hover_depth_spheres_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = quadrotor_hover_env_cfg(play=play)
  cfg.scene.env_spacing = 5.0
  cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "contact")

  for name, radius, default_pos, rgba in _OBSTACLE_SPHERE_SPECS:
    cfg.scene.entities[name] = _get_obstacle_sphere_cfg(name, radius, default_pos, rgba)

  cfg.events["sample_target"] = EventTermCfg(
    func=sample_quadrotor_target,
    mode="reset",
  )
  cfg.events["visualize_target"] = EventTermCfg(
    func=QuadrotorTargetVisualizer,
    mode="startup",
  )

  for i, name in enumerate(_SPHERE_ENTITY_NAMES):
    cfg.events[f"reset_{name}"] = EventTermCfg(
      func=sample_obstacle_sphere_near_target,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(name),
        "xy_min_radius": 1.2 + 0.08 * (i % 2),
        "xy_max_radius": 1.8 + 0.12 * (i % 3),
        "z_half_range": 0.10,
      },
    )

  depth_cam = CameraSensorCfg(
    name="depth_cam",
    camera_name="quadrotor/drone_0_camera",
    fovy=100.0,
    width=64,
    height=64,
    data_types=("depth",),
    use_shadows=False,
    use_textures=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_cam,)

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "depth_cam": ObservationTermCfg(
        func=camera_depth_obs,
        params={"sensor_name": depth_cam.name, "cutoff_distance": 8.0},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["privileged"] = ObservationGroupCfg(
    terms={
      "sphere_positions_b": ObservationTermCfg(func=sphere_positions_body_frame),
      "sphere_radii": ObservationTermCfg(func=sphere_radii_obs),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.rewards["obstacle_clearance"] = RewardTermCfg(
    func=obstacle_avoidance_penalty,
    weight=-4.0,
    params={"safety_margin": 0.7},
  )
  cfg.curriculum["obstacle_spawn"] = CurriculumTermCfg(
    func=obstacle_spawn_curriculum,
  )
  cfg.terminations["sphere_overlap"] = TerminationTermCfg(
    func=sphere_overlap_termination,
    params={"quadrotor_radius": 0.14},
  )

  if play:
    cfg.scene.num_envs = 16

  return cfg


def quadrotor_velocity_depth_spheres_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = quadrotor_hover_env_cfg(play=play)
  cfg.scene.env_spacing = 5.0
  cfg.sim.mujoco.disableflags = (*cfg.sim.mujoco.disableflags, "contact")

  for name, radius, default_pos, rgba in _OBSTACLE_SPHERE_SPECS:
    cfg.scene.entities[name] = _get_obstacle_sphere_cfg(name, radius, default_pos, rgba)

  for i, name in enumerate(_SPHERE_ENTITY_NAMES):
    cfg.events[f"reset_{name}"] = EventTermCfg(
      func=sample_obstacle_sphere_on_velocity_path,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(name),
        "x_range": (2.5 + 0.1 * (i % 2), 4.5 + 0.15 * (i % 3)),
        "y_half_range": 1.6,
        "z_range": (1.0, 2.0),
      },
    )

  depth_cam = CameraSensorCfg(
    name="depth_cam",
    camera_name="quadrotor/drone_0_camera",
    fovy=100.0,
    width=64,
    height=64,
    data_types=("depth",),
    use_shadows=False,
    use_textures=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_cam,)

  cfg.observations["actor"].terms.pop("dq_pose_error", None)
  cfg.observations["critic"].terms.pop("dq_pose_error", None)
  vel_cmd_term = ObservationTermCfg(func=desired_world_velocity)
  vel_obs_term = ObservationTermCfg(func=root_linear_velocity_world)
  cfg.observations["actor"].terms["desired_lin_vel_w"] = vel_cmd_term
  cfg.observations["actor"].terms["root_lin_vel_w"] = vel_obs_term
  cfg.observations["critic"].terms["desired_lin_vel_w"] = ObservationTermCfg(
    func=desired_world_velocity
  )
  cfg.observations["critic"].terms["root_lin_vel_w"] = ObservationTermCfg(
    func=root_linear_velocity_world
  )

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "depth_cam": ObservationTermCfg(
        func=camera_depth_obs,
        params={"sensor_name": depth_cam.name, "cutoff_distance": 8.0},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["privileged"] = ObservationGroupCfg(
    terms={
      "sphere_positions_b": ObservationTermCfg(func=sphere_positions_body_frame),
      "sphere_radii": ObservationTermCfg(func=sphere_radii_obs),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.rewards.pop("target_pose", None)
  cfg.rewards.pop("linear_velocity", None)
  cfg.rewards["world_velocity_tracking"] = RewardTermCfg(
    func=world_velocity_tracking_reward,
    weight=4.0,
    params={"desired_velocity": (1.0, 0.0, 0.0), "std": 0.4},
  )
  cfg.rewards["obstacle_clearance"] = RewardTermCfg(
    func=obstacle_avoidance_penalty,
    weight=-4.0,
    params={"safety_margin": 0.7},
  )
  cfg.curriculum["obstacle_spawn_velocity"] = CurriculumTermCfg(
    func=velocity_obstacle_curriculum,
  )
  cfg.terminations["sphere_overlap"] = TerminationTermCfg(
    func=sphere_overlap_termination,
    params={"quadrotor_radius": 0.14},
  )

  if play:
    cfg.scene.num_envs = 16

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


def quadrotor_hover_depth_spheres_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
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
    experiment_name="quadrotor_hover_depth_spheres",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=1_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera", "privileged"),
    },
  )


def quadrotor_velocity_depth_spheres_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = quadrotor_hover_depth_spheres_vision_ppo_runner_cfg()
  cfg.experiment_name = "quadrotor_velocity_depth_spheres"
  return cfg
