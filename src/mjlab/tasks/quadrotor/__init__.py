from mjlab.tasks.registry import register_mjlab_task

from .quadrotor_env_cfg import (
  quadrotor_hover_env_cfg,
  quadrotor_hover_depth_env_cfg,
  quadrotor_hover_depth_spheres_env_cfg,
  quadrotor_hover_depth_spheres_vision_ppo_runner_cfg,
  quadrotor_hover_ppo_runner_cfg,
  quadrotor_velocity_depth_spheres_env_cfg,
  quadrotor_velocity_depth_spheres_vision_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Quadrotor-Hover",
  env_cfg=quadrotor_hover_env_cfg(),
  play_env_cfg=quadrotor_hover_env_cfg(play=True),
  rl_cfg=quadrotor_hover_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Quadrotor-Hover-Depth",
  env_cfg=quadrotor_hover_depth_env_cfg(),
  play_env_cfg=quadrotor_hover_depth_env_cfg(play=True),
  rl_cfg=quadrotor_hover_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Quadrotor-Hover-Depth-Spheres",
  env_cfg=quadrotor_hover_depth_spheres_env_cfg(),
  play_env_cfg=quadrotor_hover_depth_spheres_env_cfg(play=True),
  rl_cfg=quadrotor_hover_depth_spheres_vision_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Quadrotor-Velocity-Depth-Spheres",
  env_cfg=quadrotor_velocity_depth_spheres_env_cfg(),
  play_env_cfg=quadrotor_velocity_depth_spheres_env_cfg(play=True),
  rl_cfg=quadrotor_velocity_depth_spheres_vision_ppo_runner_cfg(),
)
