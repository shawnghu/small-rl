"""Bridge to rl-rewardhacking-private repo.

Adds the repo root to sys.path so its `src.*` modules are importable.
All env wrappers that depend on that repo import this module first.

The repo is expected at ~/rl-rewardhacking-private by default.
Override with the RH_REPO_PATH environment variable.
"""

import os
import sys

RH_REPO_PATH = os.environ.get(
    "RH_REPO_PATH",
    os.path.expanduser("~/rl-rewardhacking-private"),
)


def ensure_importable():
    """Add rl-rewardhacking-private to sys.path if not already present."""
    if RH_REPO_PATH not in sys.path:
        assert os.path.isdir(RH_REPO_PATH), (
            f"rl-rewardhacking-private not found at {RH_REPO_PATH}. "
            f"Clone it there or set the RH_REPO_PATH environment variable."
        )
        sys.path.insert(0, RH_REPO_PATH)
