"""YAML config loading and merging utilities."""

import copy
import os
import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML config file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``.

    Values in ``override`` take precedence over ``base``. Nested dicts
    are merged recursively.

    Args:
        base (dict): Base configuration (e.g. from ``base_config.yaml``).
        override (dict): Override configuration (e.g. from model config).

    Returns:
        dict: Merged configuration.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
