"""Tests for Gate C1 corpus mixture configurations and validation.

Verifies schema constraints, normalized-word mixture units, top-level and
GigaVerbo residual sum invariants, pinned revision provenance, cross-mix
recipe consistency, and rejection of premature token budget fields.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pytest

from cambacica.corpus.cli import main
from cambacica.corpus.mix import (
    MixValidationError,
    load_mix_config,
    validate_mix_configs_consistency,
    validate_mix_files,
    validate_single_mix_config,
)


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root directory Path.

    Returns
    -------
    Path
        Root directory of the project.
    """
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def valid_mix_a(repo_root: Path) -> dict:
    """Load valid Candidate A mixture configuration.

    Parameters
    ----------
    repo_root : Path
        Root directory of the project.

    Returns
    -------
    dict
        Parsed Candidate A configuration dictionary.
    """
    return load_mix_config(repo_root / "configs" / "corpus_mix_a.yaml")


def test_candidate_mixes_exist_and_validate(repo_root: Path) -> None:
    """Ensure all three candidate mix YAML files exist and pass validation.

    Parameters
    ----------
    repo_root : Path
        Root directory of the project.
    """
    config_paths = [
        repo_root / "configs" / "corpus_mix_a.yaml",
        repo_root / "configs" / "corpus_mix_b.yaml",
        repo_root / "configs" / "corpus_mix_c.yaml",
    ]
    for p in config_paths:
        assert p.is_file(), f"Expected config file does not exist: {p}"

    configs = validate_mix_files(config_paths)
    assert len(configs) == 3

    assert configs[0]["name"] == "corpus_mix_a"
    assert configs[1]["name"] == "corpus_mix_b"
    assert configs[2]["name"] == "corpus_mix_c"

    # Verify exact canonical top-level weights
    assert configs[0]["sources"]["carolina"]["share"] == 0.40
    assert configs[0]["sources"]["wikipedia_pt"]["share"] == 0.25
    assert configs[0]["sources"]["parlamento_pt"]["share"] == 0.10
    assert configs[0]["sources"]["gutenberg_pt"]["share"] == 0.08
    assert configs[0]["sources"]["gigaverbo_v2_residual"]["share"] == 0.17

    assert configs[1]["sources"]["carolina"]["share"] == 0.28
    assert configs[1]["sources"]["wikipedia_pt"]["share"] == 0.18
    assert configs[1]["sources"]["parlamento_pt"]["share"] == 0.10
    assert configs[1]["sources"]["gutenberg_pt"]["share"] == 0.04
    assert configs[1]["sources"]["gigaverbo_v2_residual"]["share"] == 0.40

    assert configs[2]["sources"]["carolina"]["share"] == 0.12
    assert configs[2]["sources"]["wikipedia_pt"]["share"] == 0.07
    assert configs[2]["sources"]["parlamento_pt"]["share"] == 0.04
    assert configs[2]["sources"]["gutenberg_pt"]["share"] == 0.02
    assert configs[2]["sources"]["gigaverbo_v2_residual"]["share"] == 0.75


def test_reject_unsupported_schema_version(valid_mix_a: dict) -> None:
    """Reject configs with unsupported schema versions.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["schema_version"] = 2
    with pytest.raises(MixValidationError, match="Unsupported schema_version"):
        validate_single_mix_config(bad_cfg)


def test_reject_non_provisional_status(valid_mix_a: dict) -> None:
    """Reject configs where status is not 'provisional'.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["status"] = "frozen"
    with pytest.raises(MixValidationError, match="Expected status 'provisional'"):
        validate_single_mix_config(bad_cfg)


def test_reject_invalid_mixture_unit(valid_mix_a: dict) -> None:
    """Reject configs where mixture_unit is not 'normalized_words'.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["mixture_unit"] = "tokens"
    with pytest.raises(
        MixValidationError, match="Expected mixture_unit 'normalized_words'"
    ):
        validate_single_mix_config(bad_cfg)


def test_reject_forbidden_token_budget(valid_mix_a: dict) -> None:
    """Reject configs containing token budget fields.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["token_budget"] = 2_000_000_000
    with pytest.raises(MixValidationError, match="Forbidden token budget field"):
        validate_single_mix_config(bad_cfg)

    bad_cfg2 = copy.deepcopy(valid_mix_a)
    bad_cfg2["sources"]["carolina"]["tokens"] = 100_000_000
    with pytest.raises(MixValidationError, match="Forbidden token budget field"):
        validate_single_mix_config(bad_cfg2)


def test_reject_invalid_top_level_sum(valid_mix_a: dict) -> None:
    """Reject configs whose top-level shares do not sum to 1.0.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["sources"]["carolina"]["share"] = 0.50
    with pytest.raises(MixValidationError, match="Top-level source shares sum to"):
        validate_single_mix_config(bad_cfg)


def test_reject_negative_weight(valid_mix_a: dict) -> None:
    """Reject configs with negative source shares or subset weights.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["sources"]["carolina"]["share"] = -0.10
    with pytest.raises(MixValidationError, match="cannot be negative"):
        validate_single_mix_config(bad_cfg)

    bad_cfg2 = copy.deepcopy(valid_mix_a)
    bad_cfg2["sources"]["gigaverbo_v2_residual"]["subsets"]["quati"] = -0.05
    with pytest.raises(MixValidationError, match="cannot be negative"):
        validate_single_mix_config(bad_cfg2)


def test_reject_missing_provenance(valid_mix_a: dict) -> None:
    """Reject configs where a source has no pinned revision or snapshot provenance.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    del bad_cfg["sources"]["parlamento_pt"]["commit_sha"]
    with pytest.raises(
        MixValidationError, match="lacks a pinned revision or provenance"
    ):
        validate_single_mix_config(bad_cfg)


def test_reject_invalid_gigaverbo_residual_sum(valid_mix_a: dict) -> None:
    """Reject configs where internal GigaVerbo subset weights do not sum to 1.0.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    bad_cfg = copy.deepcopy(valid_mix_a)
    bad_cfg["sources"]["gigaverbo_v2_residual"]["subsets"]["fineweb_2_pt"] = 0.40
    with pytest.raises(
        MixValidationError, match="GigaVerbo residual internal subset weights sum to"
    ):
        validate_single_mix_config(bad_cfg)


def test_reject_cross_mix_recipe_inconsistency(valid_mix_a: dict) -> None:
    """Reject multiple mix configs that use different internal GigaVerbo recipes.

    Parameters
    ----------
    valid_mix_a : dict
        Valid baseline configuration dictionary.
    """
    cfg_a = copy.deepcopy(valid_mix_a)
    cfg_b = copy.deepcopy(valid_mix_a)

    # Modify internal subset weights in cfg_b
    cfg_b["sources"]["gigaverbo_v2_residual"]["subsets"]["finepdfs_por_Latn"] = 0.25
    cfg_b["sources"]["gigaverbo_v2_residual"]["subsets"]["fineweb_2_pt"] = 0.40

    with pytest.raises(
        MixValidationError, match="Inconsistent GigaVerbo subset weight"
    ):
        validate_mix_configs_consistency([cfg_a, cfg_b], names=["mix_a", "mix_b"])


def test_cli_validate_mixes_command(repo_root: Path) -> None:
    """Test CLI validate-mixes execution.

    Parameters
    ----------
    repo_root : Path
        Root directory of the project.
    """
    # Default files
    exit_code = main(["validate-mixes"])
    assert exit_code == 0

    # Specific file
    mix_a_str = str(repo_root / "configs" / "corpus_mix_a.yaml")
    exit_code_single = main(["validate-mixes", mix_a_str])
    assert exit_code_single == 0

    # Missing file
    exit_code_missing = main(["validate-mixes", "nonexistent_mix_file.yaml"])
    assert exit_code_missing == 1
