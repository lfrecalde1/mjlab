"""Alias entrypoint for debug_pd_actuator_viewer.py."""

from debug_pd_actuator_viewer import run, _parse_args


if __name__ == "__main__":
  run(_parse_args())
