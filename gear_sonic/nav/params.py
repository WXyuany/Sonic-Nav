from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NAV_CONFIG_DIR = REPO_ROOT / "configs" / "nav"


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    out = deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Navigation config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Navigation config must be a mapping: {config_path}")
    return data


def default_config_path(name: str) -> Path:
    if name.endswith(".yaml") or name.endswith(".yml"):
        return NAV_CONFIG_DIR / name
    return NAV_CONFIG_DIR / f"{name}.yaml"


def load_config(name: str, defaults: dict[str, Any], env_var: str | None = None) -> dict[str, Any]:
    """Load configs/nav/<name>.yaml and optionally an env-selected overlay."""

    cfg = deep_merge(defaults, load_yaml(default_config_path(name)))
    if env_var:
        cfg = deep_merge(cfg, load_yaml(os.environ.get(env_var)))
    return cfg


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def overlay_env_scalars(cfg: dict[str, Any], mapping: dict[str, tuple[str, type]]) -> dict[str, Any]:
    """Apply explicit environment overrides to a nested config copy.

    mapping values are ("path.to.key", caster). Explicit mapping keeps the
    environment API stable without guessing every possible YAML key.
    """

    out = deepcopy(cfg)
    for env_name, (path, caster) in mapping.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            value = caster(raw)
        except (TypeError, ValueError):
            continue
        cur = out
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out
