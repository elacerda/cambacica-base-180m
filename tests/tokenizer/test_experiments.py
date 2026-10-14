from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cambacica.tokenizer import experiments as c2


def test_pinned_candidate_matrix_and_config_identity() -> None:
    config = c2.load_config()
    actual = {
        (item["model_type"], item["vocab_size"])
        for item in config["candidates"]
    }
    assert actual == {
        (model_type, size)
        for model_type in ("bpe", "unigram")
        for size in (16384, 24576, 32768)
    }
    assert c2.compute_file_sha256(c2.DEFAULT_CONFIG) == c2.PINNED_CONFIG_SHA256


def test_validation_priority_is_seeded_and_namespaced() -> None:
    first = c2._priority(20261016, "doc-1", "cambacica-c2-validation-sample-v1")
    assert first == c2._priority(20261016, "doc-1", "cambacica-c2-validation-sample-v1")
    assert first != c2._priority(20261017, "doc-1", "cambacica-c2-validation-sample-v1")
    assert first != c2._priority(20261016, "doc-1", "other-sample")


def test_completed_training_stage_is_published_on_resume(tmp_path: Path) -> None:
    config = c2.load_config()
    candidate = c2.candidate_config(config, "bpe-v16384")
    identity = c2._training_identity(config, candidate, c2.PINNED_CONFIG_SHA256)
    work_dir = tmp_path / "scratch"
    work_dir.mkdir()
    (work_dir / "tokenizer.model").write_bytes(b"bounded synthetic model")
    (work_dir / "tokenizer.vocab").write_bytes(b"<unk>\n<0x00>\n")
    output_root = tmp_path / "published"
    stage = output_root / ".bpe-v16384.partial"
    stage.mkdir(parents=True)
    c2.atomic_json(stage / ".run-identity.json", identity)
    files = {}
    for name in ("tokenizer.model", "tokenizer.vocab"):
        (stage / name).write_bytes((work_dir / name).read_bytes())
        files[name] = c2.compute_file_sha256(stage / name)
    manifest = {
        "schema_version": "c2-tokenizer-artifact-v1",
        "candidate": candidate,
        "run_identity": identity,
        "files": files,
        "model": {"vocab_size": candidate["vocab_size"]},
    }
    c2.atomic_json(stage / c2.MANIFEST_NAME, manifest)
    checksums = dict(files)
    checksums[c2.MANIFEST_NAME] = c2.compute_file_sha256(stage / c2.MANIFEST_NAME)
    c2.atomic_write(
        stage / c2.CHECKSUMS_NAME,
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())).encode(),
    )

    resumed = c2._publish_training_artifact(
        work_dir,
        output_root,
        candidate,
        identity,
        {"vocab_size": candidate["vocab_size"]},
    )
    assert resumed["run_identity"] == identity
    assert (output_root / candidate["id"] / c2.MANIFEST_NAME).is_file()
    assert not stage.exists()


def test_nearest_rank_quantiles_are_fixed() -> None:
    assert c2._quantile([5, 1, 3, 2, 4], 0.50) == 3.0
    assert c2._quantile([5, 1, 3, 2, 4], 0.95) == 5.0
    assert c2._quantile([], 0.95) is None


def test_compare_writes_tables_from_synthetic_artifacts(tmp_path: Path) -> None:
    config = c2.load_config()
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    shared_identity = {
        "config_sha256": c2.PINNED_CONFIG_SHA256,
        "training_sample_manifest_sha256": c2.PINNED_TRAIN_SAMPLE_SHA256,
        "split_manifest_sha256": c2.PINNED_SPLIT_MANIFEST_SHA256,
        "split_assignments_sha256": c2.PINNED_SPLIT_ASSIGNMENTS_SHA256,
        "d1_manifest_sha256": c2.PINNED_D1_MANIFEST_SHA256,
        "evaluation_sample_sha256": "a" * 64,
        "evaluation_seed": 20261016,
    }
    for candidate in config["candidates"]:
        metrics = {
            "overall": {
                "tokens_per_normalized_word": 1.5,
                "tokens_per_word_including_eos": 1.6,
                "tokens_per_unicode_codepoint": 0.4,
                "tokens_per_word_distribution": {"p50": 1, "p95": 3, "p99": 4},
                "accented_portuguese_words": {
                    "mean_tokens": 1.2,
                    "single_piece_rate": 0.5,
                },
                "unknown_token_rate": 0.0,
                "byte_fallback_token_rate": 0.02,
                "numbers": {
                    "mean_tokens_per_span": 1.5,
                    "scientific_mean_tokens_per_span": 2.0,
                },
                "punctuation": {"single_piece_rate": 0.9},
                "round_trip_exact_documents": 4,
                "round_trip_mismatched_documents": 0,
                "round_trip_exact_rate": 1.0,
                "documents_per_second": 100.0,
                "characters_per_second": 10000.0,
                "tokens_per_second": 15000.0,
            },
            "source_balanced_macro": {
                "metrics": {
                    "tokens_per_normalized_word": 1.5,
                    "accented_portuguese_words_mean_tokens": 1.2,
                    "rare_sample_words_single_piece_type_rate": 0.5,
                    "technical_lexicon_single_piece_type_rate": 0.6,
                    "word_p50": 1,
                    "word_p95": 3,
                    "word_p99": 4,
                }
            },
            "model_parameter_cost": {
                "reference_hidden_width": 768,
                "embedding_matrix_parameters": candidate["vocab_size"] * 768,
                "output_head_parameters_if_untied": candidate["vocab_size"] * 768,
            },
            "tokenizer_artifact_bytes": 100,
            "by_source": {
                "carolina": {
                    "documents": 4,
                    "normalized_words": 100,
                    "tokens_per_normalized_word": 1.5,
                    "tokens_per_unicode_codepoint": 0.4,
                    "unknown_token_rate": 0.0,
                    "byte_fallback_token_rate": 0.02,
                    "tokens_per_word_distribution": {"p50": 1, "p95": 3, "p99": 4},
                    "accented_portuguese_words": {
                        "mean_tokens": 1.2,
                        "single_piece_rate": 0.5,
                    },
                    "rare_sample_words": {
                        "single_piece_type_rate": 0.5,
                        "mean_tokens_per_occurrence": 2.0,
                        "byte_fallback_piece_rate": 0.02,
                    },
                    "technical_lexicon": {
                        "single_piece_type_rate": 0.6,
                        "mean_tokens_per_occurrence": 2.2,
                        "byte_fallback_piece_rate": 0.01,
                    },
                    "numbers": {
                        "spans": 2,
                        "mean_tokens_per_span": 1.5,
                        "scientific_notation_spans": 1,
                        "scientific_mean_tokens_per_span": 2.0,
                    },
                    "punctuation": {"single_piece_rate": 0.9},
                    "round_trip": {"exact_rate": 1.0},
                    "documents_per_second": 100.0,
                    "characters_per_second": 10000.0,
                    "tokens_per_second": 15000.0,
                }
            },
            "by_source_subset_domain": {
                "carolina/wik/wik": {
                    "documents": 4,
                    "normalized_words": 100,
                    "unicode_codepoints": 400,
                    "tokens_per_normalized_word": 1.5,
                    "tokens_per_unicode_codepoint": 0.4,
                    "unknown_token_rate": 0.0,
                    "byte_fallback_token_rate": 0.02,
                    "tokens_per_word_distribution": {"p50": 1, "p95": 3, "p99": 4},
                    "accented_portuguese_words": {
                        "mean_tokens": 1.2,
                        "single_piece_rate": 0.5,
                    },
                    "rare_sample_words": {
                        "single_piece_type_rate": 0.5,
                        "mean_tokens_per_occurrence": 2.0,
                        "byte_fallback_piece_rate": 0.02,
                    },
                    "technical_lexicon": {
                        "single_piece_type_rate": 0.6,
                        "mean_tokens_per_occurrence": 2.2,
                        "byte_fallback_piece_rate": 0.01,
                    },
                    "numbers": {
                        "spans": 2,
                        "mean_tokens_per_span": 1.5,
                        "scientific_notation_spans": 1,
                        "scientific_mean_tokens_per_span": 2.0,
                    },
                    "punctuation": {"characters": 10, "single_piece_rate": 0.9},
                    "round_trip": {"exact_rate": 1.0},
                    "documents_per_second": 100.0,
                    "characters_per_second": 10000.0,
                    "tokens_per_second": 15000.0,
                }
            },
        }
        payload = {
            "candidate": candidate,
            "evaluation_identity": {
                **shared_identity,
                "tokenizer_model_sha256": "b" * 64,
            },
            "metrics": metrics,
        }
        (evaluation_root / f"{candidate['id']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    output_dir = tmp_path / "comparison"
    result = c2.compare_evaluations(
        evaluation_root, output_dir, require_nfs=False
    )
    assert result["winner_declared"] is False
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["accented_word_single_piece_rate"] == "0.5"
    assert rows[0]["unknown_token_rate"] == "0.0"
    assert rows[0]["embedding_matrix_parameters"] == str(16384 * 768)
    with (output_dir / "by_source.csv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(source_rows) == 6
    assert source_rows[0]["source"] == "carolina"
    with (output_dir / "by_source_subset_domain.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        stratum_rows = list(csv.DictReader(stream))
    assert len(stratum_rows) == 6
    assert stratum_rows[0]["source_subset_domain"] == "carolina/wik/wik"
    manifest = json.loads((output_dir / "comparison_manifest.json").read_text())
    assert manifest["candidate_count"] == 6
    assert manifest["winner_declared"] is False


@pytest.mark.parametrize("model_type", ["bpe", "unigram"])
def test_synthetic_sentencepiece_training_round_trips(model_type: str, tmp_path: Path) -> None:
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "tiny.txt"
    line = "ação portuguesa; número 1.25e-4, ciência e dados."
    corpus.write_text((line + "\n") * 120, encoding="utf-8")
    prefix = tmp_path / model_type
    spm.set_random_generator_seed(20261015)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type=model_type,
        vocab_size=300,
        hard_vocab_limit=False,
        character_coverage=1.0,
        byte_fallback=True,
        num_threads=1,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        escape_whitespaces=True,
        split_by_unicode_script=True,
        split_by_number=False,
        split_digits=False,
        split_by_whitespace=True,
        treat_whitespace_as_suffix=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        seed_sentencepiece_size=1000,
        minloglevel=2,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    assert processor.unk_id() == 0
    assert processor.bos_id() == 1
    assert processor.eos_id() == 2
    assert processor.pad_id() == -1
    probe = "ação  e-mail\t3.14e+5!\nCiência."
    assert processor.decode(processor.encode(probe, out_type=int)) == probe
