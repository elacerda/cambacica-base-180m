"""Validation helper for Gate C1 corpus mixture configurations.

Provides functions to load and validate provisional corpus mixture definitions,
enforcing scientific invariants such as normalized-word units, floating-point
weight sums, pinned revision provenance, unified GigaVerbo residual recipes,
and absence of premature token budgets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import yaml

SUPPORTED_SCHEMA_VERSIONS = {1}
REQUIRED_STATUS = "provisional"
REQUIRED_MIXTURE_UNIT = "normalized_words"
FLOAT_TOLERANCE = 1e-6

FORBIDDEN_TOKEN_KEYS = {
    "token_budget",
    "tokens",
    "num_tokens",
    "total_tokens",
    "final_tokens",
    "budget",
    "target_tokens",
}


class MixValidationError(ValueError):
    """Exception raised when a corpus mix configuration fails validation."""

    pass


def load_mix_config(config_path: Union[Path, str]) -> Dict[str, Any]:
    """Load and parse a YAML corpus mixture configuration file.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML mixture configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    MixValidationError
        If the file does not exist or does not contain a valid mapping.
    """
    path = Path(config_path)
    if not path.is_file():
        raise MixValidationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise MixValidationError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MixValidationError(
            f"Expected YAML file {path} to contain a mapping/dictionary, got {type(data).__name__}"
        )

    return data


def _check_no_forbidden_keys(data: Any, path_prefix: str = "") -> None:
    """Recursively verify that no premature token budget fields are present.

    Parameters
    ----------
    data : Any
        Arbitrary data structure to traverse.
    path_prefix : str, default=""
        Current path prefix for error messaging.

    Raises
    ------
    MixValidationError
        If any forbidden token budget key is detected.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else str(key)
            key_str = str(key).lower()
            if (
                key_str in FORBIDDEN_TOKEN_KEYS
                or key_str.endswith("_tokens")
                or key_str.endswith("_token_budget")
            ):
                raise MixValidationError(
                    f"Forbidden token budget field detected at '{current_path}'. "
                    f"Gate C1 mixtures must NOT contain a final token budget."
                )
            _check_no_forbidden_keys(value, current_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _check_no_forbidden_keys(item, f"{path_prefix}[{idx}]")


def validate_single_mix_config(
    config: Dict[str, Any], config_name: str = "config"
) -> None:
    """Validate a single Gate C1 corpus mixture configuration dictionary.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary to validate.
    config_name : str, default="config"
        Human-readable name or filename for diagnostic error messages.

    Raises
    ------
    MixValidationError
        If any validation rule is violated.
    """
    # 1. Schema version
    schema_version = config.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise MixValidationError(
            f"[{config_name}] Unsupported schema_version '{schema_version}'. "
            f"Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    # 2. Gate
    gate = config.get("gate")
    if gate != "C1":
        raise MixValidationError(f"[{config_name}] Expected gate 'C1', found '{gate}'.")

    # 3. Status must be provisional
    status = config.get("status")
    if status != REQUIRED_STATUS:
        raise MixValidationError(
            f"[{config_name}] Expected status '{REQUIRED_STATUS}', found '{status}'."
        )

    # 4. Mixture unit must be normalized_words
    mixture_unit = config.get("mixture_unit")
    if mixture_unit != REQUIRED_MIXTURE_UNIT:
        raise MixValidationError(
            f"[{config_name}] Expected mixture_unit '{REQUIRED_MIXTURE_UNIT}', found '{mixture_unit}'."
        )

    # 5. No token budget fields allowed
    _check_no_forbidden_keys(config)

    # 6. Sources mapping
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise MixValidationError(
            f"[{config_name}] 'sources' must be a non-empty dictionary."
        )

    top_level_sum = 0.0
    for src_name, src_info in sources.items():
        if not isinstance(src_info, dict):
            raise MixValidationError(
                f"[{config_name}] Source '{src_name}' configuration must be a dictionary."
            )

        # Share validation
        share = src_info.get("share")
        if share is None or not isinstance(share, (int, float)):
            raise MixValidationError(
                f"[{config_name}] Source '{src_name}' must have a numeric 'share'."
            )
        if share < 0.0:
            raise MixValidationError(
                f"[{config_name}] Source '{src_name}' share cannot be negative: {share}."
            )
        top_level_sum += float(share)

        # Provenance / revision check
        has_commit = bool(src_info.get("commit_sha"))
        has_revision = bool(src_info.get("revision"))
        has_snapshot = bool(
            src_info.get("snapshot_date") or src_info.get("snapshot_policy")
        )

        if not (has_commit or has_revision or has_snapshot):
            raise MixValidationError(
                f"[{config_name}] Source '{src_name}' lacks a pinned revision or provenance "
                f"identifier (requires commit_sha, revision, or snapshot_date/policy)."
            )

        # Check internal GigaVerbo residual recipe if applicable
        if src_name == "gigaverbo_v2_residual":
            subsets = src_info.get("subsets")
            if not isinstance(subsets, dict) or not subsets:
                raise MixValidationError(
                    f"[{config_name}] Source '{src_name}' must define a non-empty 'subsets' dictionary."
                )

            subset_sum = 0.0
            for sub_name, sub_weight in subsets.items():
                if not isinstance(sub_weight, (int, float)):
                    raise MixValidationError(
                        f"[{config_name}] Subset '{sub_name}' in '{src_name}' must have a numeric weight."
                    )
                if sub_weight < 0.0:
                    raise MixValidationError(
                        f"[{config_name}] Subset '{sub_name}' weight cannot be negative: {sub_weight}."
                    )
                subset_sum += float(sub_weight)

            if not math.isclose(subset_sum, 1.0, abs_tol=FLOAT_TOLERANCE):
                raise MixValidationError(
                    f"[{config_name}] GigaVerbo residual internal subset weights sum to "
                    f"{subset_sum:.8f}, expected exactly 1.0 (tolerance: {FLOAT_TOLERANCE})."
                )

    # Check top-level shares sum
    if not math.isclose(top_level_sum, 1.0, abs_tol=FLOAT_TOLERANCE):
        raise MixValidationError(
            f"[{config_name}] Top-level source shares sum to {top_level_sum:.8f}, "
            f"expected exactly 1.0 (tolerance: {FLOAT_TOLERANCE})."
        )


def validate_mix_configs_consistency(
    configs: Sequence[Dict[str, Any]], names: Optional[Sequence[str]] = None
) -> None:
    """Verify that multiple candidate mix configurations share identical residual policies.

    Specifically, ensures all candidate configs define the identical set of
    GigaVerbo-v2 residual subsets with identical internal weights.

    Parameters
    ----------
    configs : sequence of dict
        Sequence of parsed configuration dictionaries.
    names : sequence of str or None, optional
        Names or filepaths corresponding to configs, for error diagnostics.

    Raises
    ------
    MixValidationError
        If GigaVerbo residual recipes differ between configurations.
    """
    if len(configs) <= 1:
        return

    reference_recipe: Optional[Dict[str, float]] = None
    reference_name: str = ""

    for idx, cfg in enumerate(configs):
        name = names[idx] if names and idx < len(names) else f"config_{idx}"
        sources = cfg.get("sources", {})
        residual = sources.get("gigaverbo_v2_residual", {})
        subsets = residual.get("subsets")

        if subsets is None:
            continue

        normalized_recipe = {k: float(v) for k, v in subsets.items()}

        if reference_recipe is None:
            reference_recipe = normalized_recipe
            reference_name = name
        else:
            if set(normalized_recipe.keys()) != set(reference_recipe.keys()):
                raise MixValidationError(
                    f"Inconsistent GigaVerbo residual subsets between '{reference_name}' "
                    f"and '{name}': {set(reference_recipe.keys()) ^ set(normalized_recipe.keys())}"
                )
            for sub_k, sub_v in reference_recipe.items():
                cur_v = normalized_recipe[sub_k]
                if not math.isclose(sub_v, cur_v, abs_tol=FLOAT_TOLERANCE):
                    raise MixValidationError(
                        f"Inconsistent GigaVerbo subset weight for '{sub_k}' between "
                        f"'{reference_name}' ({sub_v}) and '{name}' ({cur_v})."
                    )


def validate_mix_files(file_paths: Sequence[Union[Path, str]]) -> List[Dict[str, Any]]:
    """Load, validate, and check cross-consistency for a sequence of mixture YAML files.

    Parameters
    ----------
    file_paths : sequence of Path or str
        Paths to mixture YAML files to validate.

    Returns
    -------
    list of dict
        List of parsed and validated configuration dictionaries.

    Raises
    ------
    MixValidationError
        If any individual file or cross-consistency validation fails.
    """
    loaded_configs: List[Dict[str, Any]] = []
    file_names: List[str] = []

    for f_path in file_paths:
        str_path = str(f_path)
        cfg = load_mix_config(f_path)
        validate_single_mix_config(cfg, config_name=str_path)
        loaded_configs.append(cfg)
        file_names.append(str_path)

    validate_mix_configs_consistency(loaded_configs, names=file_names)
    return loaded_configs
