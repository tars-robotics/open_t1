# Copyright (c) 2026 TARS
# SPDX-License-Identifier: MIT

"""Export a trained T1 policy checkpoint to ONNX.

Thin wrapper around ``scripts/rsl_rl/play.py --export_only`` that, before any
import of ``rsl_rl``, reorders the two editable ``rsl_rl`` import finders so the
``rsl_rl`` >= 5.5.0 install wins over a stale 2.3.3 one. This is process-local
only -- it does not modify the active conda env.

Usage:
    python scripts/rsl_rl/export_onnx.py --task=Tracking-Flat-T1-v0 \
        --load_path /path/to/model_1500.pt --motion_file /path/to/motion.npz \
        --export_only --headless
"""

import os
import re
import sys


def _module_of(finder):
    return getattr(finder, "__module__", "")


def _prefer_newer_rsl_rl():
    """Ensure the highest-version editable ``rsl_rl`` finder is searched first on ``sys.meta_path``.

    Each editable ``rsl_rl`` install registers a finder named with its version
    (e.g. ``__editable___rsl_rl_lib_2_3_3_finder``). A stale 2.x install would otherwise
    shadow the 5.x one, since both provide the top-level ``rsl_rl`` package. Reorder the
    finders by version so the newest one wins, regardless of the exact version strings.
    """
    finder_versions = []
    for finder in sys.meta_path:
        match = re.fullmatch(r"__editable___rsl_rl_lib_(?P<version>\d+(?:_\d+)*)_finder", _module_of(finder))
        if match:
            version = tuple(int(part) for part in match.group("version").split("_"))
            finder_versions.append((version, finder))
    if len(finder_versions) < 2:
        return
    finder_versions.sort(key=lambda item: item[0], reverse=True)
    newest_finder = finder_versions[0][1]
    sys.meta_path.remove(newest_finder)
    insert_at = min(sys.meta_path.index(f) for _, f in finder_versions[1:])
    sys.meta_path.insert(insert_at, newest_finder)


_prefer_newer_rsl_rl()

# Delegate to play.py (its module-level code launches Isaac Sim and runs the export).
import runpy

_play_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play.py")
sys.argv = [_play_path] + sys.argv[1:]
runpy.run_path(_play_path, run_name="__main__")