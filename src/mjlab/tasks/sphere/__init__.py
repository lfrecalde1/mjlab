from mjlab.tasks.registry import register_mjlab_task

from .sphere_env_cfg import sphere_translate_env_cfg, sphere_translate_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Sphere-Translate",
  env_cfg=sphere_translate_env_cfg(),
  play_env_cfg=sphere_translate_env_cfg(play=True),
  rl_cfg=sphere_translate_ppo_runner_cfg(),
)

