"""Configuration loading.

Every tunable in this project lives in ``conf/config.yaml``. Modules receive a
``Config`` object rather than reading globals, so the same transformation code
runs unchanged against a local Parquet lake or a Databricks Delta catalogue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Thin, dotted-access wrapper over the YAML configuration."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value with ``a.b.c`` syntax."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"No path configured at '{dotted}'")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``conf/config.yaml`` (or an explicit override)."""
    root = project_root()
    cfg_path = Path(path) if path else root / "conf" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=raw, root=root)
