"""
Compatibility fix loaded by every Python process of a vLLM server whose catalog `env` puts this folder on PYTHONPATH
(Python imports `sitecustomize` at startup; vLLM's engine and model-inspection subprocesses inherit the variable).

vLLM 0.29.0's pixtral.py imports PixtralRotaryEmbedding and position_ids_in_meshgrid from transformers, which
transformers 5.17 no longer has, so every Pixtral/Mistral 3 model fails to load. Only the HF-format vision tower
(PixtralHFVisionModel) calls them; mistral-format weights (--config-format mistral) use vLLM's own VisionTransformer.
Adding stand-ins that raise if called lets the module import. Missing names only: a no-op once transformers or vLLM
is fixed, after which this folder can be dropped from the catalog.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from typing import Iterable

PIXTRAL = "transformers.models.pixtral.modeling_pixtral"
PIXTRAL_REMOVED = ("PixtralRotaryEmbedding", "position_ids_in_meshgrid")


def _removed(module: str, name: str):
    def stand_in(*args, **kwargs):
        raise NotImplementedError(f"{module}.{name} does not exist in this transformers version (vllm_compat stand-in)")

    return stand_in


class AliasFinder(importlib.abc.MetaPathFinder):
    """Right after module `target` is imported, gives it a raising stand-in for each of `names` it lacks."""

    def __init__(self, target: str, names: Iterable[str]):
        self.target, self.names = target, tuple(names)

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            for missing in self.names:
                if not hasattr(module, missing):
                    setattr(module, missing, _removed(self.target, missing))

        spec.loader.exec_module = exec_module
        return spec


if __name__ == "sitecustomize":
    sys.meta_path.insert(0, AliasFinder(PIXTRAL, PIXTRAL_REMOVED))
