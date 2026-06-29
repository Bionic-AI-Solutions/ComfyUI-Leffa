# -*- coding: utf-8 -*-
"""ComfyUI-Leffa — full-pipeline virtual try-on & pose transfer (Meta Leffa).

The vendored upstream packages (`leffa`, `leffa_utils`, `preprocess`,
`detectron2`, `densepose`, `SCHP`) use absolute, top-level imports and expect to
be run from the repo root. ComfyUI imports this folder as a package, so we insert
the folder onto `sys.path` here to make those top-level imports resolve.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
