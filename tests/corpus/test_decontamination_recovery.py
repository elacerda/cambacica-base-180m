from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cambacica.corpus.decontamination.matcher import (
    CHECKPOINT_SCHEMA_VERSION,
    MATCHER_VERSION,
    NORMALIZATION_VERSION,
    BenchmarkMatcher,
    CandidatePolicy,
    MatchField,
    _anchor_digest,
)
from cambacica.corpus.decontamination.production import (
    ACCOUNTING_SCHEMA,
    SCAN_VERSION,
    _canonical_sha256,
    _corpus_documents,
    _publish_prepared_stage,
    _verify_output_records,
    _write_anchor_evidence,
    _write_hits,
)
from cambacica.corpus.decontamination import production as production_module
from cambacica.corpus.decontamination.snapshot import canonical_json, sha256_file


def _field(example_id: str, role: str, text: str) -> MatchField:
    return MatchField(
        benchmark_name="synthetic_benchmark",
        example_id=example_id,
        source_row_id=example_id,
        source_file_sha256=hashlib.sha256(b"synthetic-snapshot-source").hexdigest(),
        source_category="synthetic",
        field_id=f"{example_id}:{role}",
        field_role=role,
        original_text=text,
    )


def _make_corpus(root: Path) -> tuple[list[dict[str, object]], str, list[MatchField]]:
    passage_tokens = [
        "a",
        "comunidade",
        "costeira",
        "registrou",
        "as",
        "mareas",
        "durante",
        "o",
        "inverno",
        "em",
        "um",
        "observatorio",
        "construido",
        "perto",
        "da",
        "enseada",
        "e",
        "comparou",
        "as",
        "medicoes",
        "com",
        "relatos",
        "de",
        "pescadores",
        "que",
        "viviam",
        "na",
        "regiao",
        "ha",
        "muitas",
        "decadas",
        "os",
        "dados",
        "foram",
        "organizados",
        "em",
        "cadernos",
        "publicos",
        "para",
        "apoiar",
        "a",
        "navegacao",
        "local",
        "e",
        "a",
        "preservacao",
        "dos",
        "manguezais",
        "que",
        "protegem",
        "a",
        "praia",
    ] + [f"termo{index:04d}" for index in range(75)]
    passage = " ".join(passage_tokens)
    question = (
        "Qual observatorio organizou as medicoes costeiras durante o inverno "
        "na enseada de Pedra Clara?"
    )
    documents = [
        [
            f"Registro de arquivo. {passage} Encerramento.",
            "Texto neutro sobre um acervo municipal sem trecho de benchmark.",
            passage,
        ],
        [
            passage,
            "Relatorio de campo sem sobreposicao lexical relevante.",
            f"Pergunta arquivada: {question}",
        ],
        [
            " ".join(passage_tokens[:55]) + " notas complementares sem copia integral.",
            passage,
            "Descricao independente de um arquivo regional e suas catalogacoes.",
        ],
    ]
    root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    for shard_index, rows in enumerate(documents):
        relative = f"data/familia-{shard_index // 2}/part-{shard_index:02d}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "text": rows,
                "source": [f"familia-{shard_index // 2}"] * len(rows),
                "content_sha256": [
                    hashlib.sha256(text.encode("utf-8")).hexdigest() for text in rows
                ],
            }
        )
        pq.write_table(table, path, row_group_size=2, compression="zstd")
        inventory.append(
            {
                "relative": relative,
                "normalized_shard": Path(relative).relative_to("data").as_posix(),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_identity = hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest()
    fields = [
        _field("reading-1", "passage", passage),
        _field("reading-1", "question", question),
    ]
    return inventory, manifest_identity, fields


def _identity(
    inventory: list[dict[str, object]],
    manifest_identity: str,
    policy: CandidatePolicy,
    **overrides: str,
) -> dict[str, object]:
    policy_json = canonical_json(policy.to_dict())
    identity: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": "synthetic-recovery-test",
        "output_dir": "/synthetic/output",
        "d1_corpus_manifest_sha256": manifest_identity,
        "ordered_source_shard_inventory": inventory,
        "ordered_source_shard_inventory_sha256": _canonical_sha256(inventory),
        "benchmark_snapshot_manifest_sha256": "a" * 64,
        "benchmark_calibration_manifest_sha256": "b" * 64,
        "matcher_policy_sha256": hashlib.sha256(
            policy_json.encode("utf-8")
        ).hexdigest(),
        "matcher_policy": policy.to_dict(),
        "matcher_implementation_version": MATCHER_VERSION,
        "normalization_implementation_version": NORMALIZATION_VERSION,
    }
    identity.update(overrides)
    return identity


def _matcher(policy: CandidatePolicy, fields: list[MatchField]) -> BenchmarkMatcher:
    return BenchmarkMatcher(fields, policy)


def _open_run(
    matcher: BenchmarkMatcher,
    db_path: Path,
    identity: dict[str, object],
    inventory: list[dict[str, object]],
    *,
    resume: bool,
):
    return matcher.start_run(
        database_path=db_path,
        resume=resume,
        checkpoint_identity=identity,
        shard_inventory=inventory,
    )


def _remaining_documents(
    root: Path,
    inventory: list[dict[str, object]],
    manifest_identity: str,
    run,
):
    position = run.resume_position
    return _corpus_documents(
        root,
        inventory,
        manifest_identity,
        start_shard_index=position["shard_index"],
        start_row_ordinal=position["row_ordinal"],
    )


def _finish_from_checkpoint(
    root: Path,
    inventory: list[dict[str, object]],
    manifest_identity: str,
    matcher: BenchmarkMatcher,
    db_path: Path,
    identity: dict[str, object],
    output: Path,
    *,
    resume: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    with _open_run(matcher, db_path, identity, inventory, resume=resume) as run:
        if run.checkpoint_status not in {"FINALIZED", "PUBLISHED"}:
            run.scan(_remaining_documents(root, inventory, manifest_identity, run))
        else:
            run.finish()
        results = [asdict(item) for item in run.iter_results()]
        evidence = [asdict(item) for item in run.iter_anchor_evidence()]
        accounting = [asdict(item) for item in run.finish().shard_accounting]
        frequencies = run.anchor_frequencies()
        _write_hits(output / "candidate_hits.parquet", run.iter_results())
        _write_anchor_evidence(
            output / "candidate_anchor_evidence.parquet", run.iter_anchor_evidence()
        )
        account_rows = []
        by_shard = {item["source_shard"]: item for item in accounting}
        result_counts: dict[str, int] = {}
        for result in results:
            shard = result["source_shard"]
            result_counts[shard] = result_counts.get(shard, 0) + 1
        for item in inventory:
            shard = str(item["normalized_shard"])
            shard_account = by_shard[shard]
            account_rows.append(
                {
                    "normalized_shard": shard,
                    "expected_rows": item["rows"],
                    "accounted_rows": shard_account["document_count"],
                    "record_id_sha256": shard_account["record_id_sha256"],
                    "candidate_rows": result_counts.get(shard, 0),
                }
            )
        pq.write_table(
            pa.Table.from_pylist(account_rows, schema=ACCOUNTING_SCHEMA),
            output / "scan_accounting.parquet",
            compression="zstd",
            version="2.6",
        )
    return {
        "results": results,
        "evidence": evidence,
        "accounting": accounting,
        "frequencies": frequencies,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in sorted(output.glob("*.parquet"))
        },
    }


def _hard_kill_after_documents(
    root: Path,
    inventory: list[dict[str, object]],
    manifest_identity: str,
    policy: CandidatePolicy,
    fields: list[MatchField],
    db_path: Path,
    identity: dict[str, object],
    stop_after: int,
) -> None:
    matcher = _matcher(policy, fields)
    run = _open_run(matcher, db_path, identity, inventory, resume=db_path.exists())
    count = 0
    for document in _remaining_documents(root, inventory, manifest_identity, run):
        run.add_document(document)
        count += 1
        if count == stop_after:
            os.kill(os.getpid(), signal.SIGKILL)
    run.close()


def _checkpoint_state(
    matcher: BenchmarkMatcher,
    db_path: Path,
    identity: dict[str, object],
    inventory: list[dict[str, object]],
) -> tuple[str, dict[str, int]]:
    with _open_run(matcher, db_path, identity, inventory, resume=True) as run:
        return run.checkpoint_status, run.resume_position


def test_hard_kill_and_resume_match_uninterrupted_artifacts(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=2, max_ngram_df=2)
    identity = _identity(inventory, manifest_identity, policy)
    baseline = _finish_from_checkpoint(
        root,
        inventory,
        manifest_identity,
        _matcher(policy, fields),
        tmp_path / "baseline.sqlite3",
        identity,
        tmp_path / "baseline-output",
        resume=False,
    )

    database = tmp_path / "resumed.sqlite3"
    process = multiprocessing.get_context("fork").Process(
        target=_hard_kill_after_documents,
        args=(
            root,
            inventory,
            manifest_identity,
            policy,
            fields,
            database,
            identity,
            2,
        ),
    )
    process.start()
    process.join(20)
    assert process.exitcode == -signal.SIGKILL
    status, position = _checkpoint_state(
        _matcher(policy, fields), database, identity, inventory
    )
    assert status == "SCANNING"
    assert position == {"shard_index": 0, "row_ordinal": 2, "documents_seen": 2}

    resumed = _finish_from_checkpoint(
        root,
        inventory,
        manifest_identity,
        _matcher(policy, fields),
        database,
        identity,
        tmp_path / "resumed-output",
        resume=True,
    )
    assert resumed["results"] == baseline["results"]
    assert resumed["evidence"] == baseline["evidence"]
    assert resumed["accounting"] == baseline["accounting"]
    assert resumed["frequencies"] == baseline["frequencies"]
    assert resumed["artifact_sha256"] == baseline["artifact_sha256"]


def test_corpus_reader_ordinals_continue_across_batches_in_a_row_group(tmp_path):
    root = tmp_path / "corpus"
    path = root / "data" / "family" / "part-00.parquet"
    path.parent.mkdir(parents=True)
    texts = [f"registro de teste numero {index}" for index in range(23)]
    table = pa.table(
        {
            "text": texts,
            "source": ["family"] * len(texts),
            "content_sha256": [
                hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
            ],
        }
    )
    pq.write_table(table, path, row_group_size=23, compression="zstd")
    inventory = [
        {
            "relative": "data/family/part-00.parquet",
            "normalized_shard": "family/part-00.parquet",
            "rows": len(texts),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    ]

    documents = list(
        _corpus_documents(
            root,
            inventory,
            "a" * 64,
            batch_size=7,
        )
    )

    assert [document.source_row_ordinal for document in documents] == list(
        range(len(texts))
    )
    assert len({document.doc_id for document in documents}) == len(texts)
    assert [document.text for document in documents] == texts
    resumed_documents = list(
        _corpus_documents(
            root,
            inventory,
            "a" * 64,
            batch_size=7,
            start_shard_index=0,
            start_row_ordinal=17,
        )
    )
    assert [document.source_row_ordinal for document in resumed_documents] == list(
        range(17, len(texts))
    )
    assert [document.text for document in resumed_documents] == texts[17:]


def test_hard_kill_before_first_checkpoint_replays_only_uncommitted_rows(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=3)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    process = multiprocessing.get_context("fork").Process(
        target=_hard_kill_after_documents,
        args=(
            root,
            inventory,
            manifest_identity,
            policy,
            fields,
            database,
            identity,
            2,
        ),
    )
    process.start()
    process.join(20)
    assert process.exitcode == -signal.SIGKILL
    _status, position = _checkpoint_state(
        _matcher(policy, fields), database, identity, inventory
    )
    assert position == {"shard_index": 0, "row_ordinal": 0, "documents_seen": 0}
    completed = _finish_from_checkpoint(
        root,
        inventory,
        manifest_identity,
        _matcher(policy, fields),
        database,
        identity,
        tmp_path / "output",
        resume=True,
    )
    assert len(completed["accounting"]) == len(inventory)


def test_hard_kill_immediately_after_checkpoint_between_shards(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=3)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    process = multiprocessing.get_context("fork").Process(
        target=_hard_kill_after_documents,
        args=(
            root,
            inventory,
            manifest_identity,
            policy,
            fields,
            database,
            identity,
            3,
        ),
    )
    process.start()
    process.join(20)
    assert process.exitcode == -signal.SIGKILL
    _status, position = _checkpoint_state(
        _matcher(policy, fields), database, identity, inventory
    )
    assert position == {"shard_index": 1, "row_ordinal": 0, "documents_seen": 3}


def test_repeated_interruptions_and_multiple_resumes_keep_one_accounting_pass(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=3)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"

    for stop_after, expected_committed in ((2, 0), (3, 3), (2, 3)):
        process = multiprocessing.get_context("fork").Process(
            target=_hard_kill_after_documents,
            args=(
                root,
                inventory,
                manifest_identity,
                policy,
                fields,
                database,
                identity,
                stop_after,
            ),
        )
        process.start()
        process.join(20)
        assert process.exitcode == -signal.SIGKILL
        _status, position = _checkpoint_state(
            _matcher(policy, fields), database, identity, inventory
        )
        assert position["documents_seen"] == expected_committed

    completed = _finish_from_checkpoint(
        root,
        inventory,
        manifest_identity,
        _matcher(policy, fields),
        database,
        identity,
        tmp_path / "output",
        resume=True,
    )
    assert sum(item["document_count"] for item in completed["accounting"]) == 9


def test_interrupted_finalization_rolls_back_and_restarts_idempotently(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=2)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    matcher = _matcher(policy, fields)
    with _open_run(matcher, database, identity, inventory, resume=False) as run:
        for document in _remaining_documents(root, inventory, manifest_identity, run):
            run.add_document(document)
        original_insert = run._insert_result
        inserted = 0

        def fail_after_one(result):
            nonlocal inserted
            original_insert(result)
            inserted += 1
            if inserted == 1:
                raise RuntimeError("injected finalization interruption")

        run._insert_result = fail_after_one
        with pytest.raises(RuntimeError, match="finalization interruption"):
            run.finish()
        assert run.checkpoint_status == "FINALIZING"

    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=True
    ) as run:
        accounting = run.finish()
        first_results = [asdict(item) for item in run.iter_results()]
        assert run.checkpoint_status == "FINALIZED"
        assert run.finish() == accounting
        assert [asdict(item) for item in run.iter_results()] == first_results
        assert accounting.documents_seen == 9


def test_global_anchor_frequency_crosses_ceiling_after_resume_and_exact_survives(
    tmp_path,
):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=1, max_ngram_df=2)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    matcher = _matcher(policy, fields)
    with _open_run(matcher, database, identity, inventory, resume=False) as run:
        documents = _remaining_documents(root, inventory, manifest_identity, run)
        for index, document in enumerate(documents):
            run.add_document(document)
            if index == 2:
                break
        assert run.resume_position["documents_seen"] == 3
        passage_tokens = tuple(matcher.field_token_values[fields[0].field_id][:13])
        anchor = _anchor_digest(passage_tokens)
        assert run.anchor_frequencies()[anchor] == 2
        assert (
            run.connection.execute(
                "SELECT count(*) FROM anchor_evidence WHERE anchor=?", (anchor,)
            ).fetchone()[0]
            > 0
        )

    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=True
    ) as run:
        run.scan(_remaining_documents(root, inventory, manifest_identity, run))
        anchor = _anchor_digest(
            tuple(run.matcher.field_token_values[fields[0].field_id][:13])
        )
        assert run.anchor_frequencies()[anchor] == 5
        assert (
            run.connection.execute(
                "SELECT frequent FROM anchor_frequency WHERE anchor=?", (anchor,)
            ).fetchone()[0]
            == 1
        )
        assert (
            run.connection.execute(
                "SELECT count(*) FROM anchor_evidence WHERE anchor=?", (anchor,)
            ).fetchone()[0]
            == 0
        )
        exact_hits = [
            hit
            for hit in run.iter_results()
            if hit.field_role == "passage" and hit.decision_rule == "exact_passage"
        ]
        assert len(exact_hits) == 4
        assert all(hit.exact_match for hit in exact_hits)
        assert all(
            evidence.anchor_sha256 != anchor for evidence in run.iter_anchor_evidence()
        )


def test_incompatible_corpus_manifest_refuses_resume(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ):
        pass
    changed = dict(identity, d1_corpus_manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="identity does not match"):
        _open_run(_matcher(policy, fields), database, changed, inventory, resume=True)


def test_incompatible_benchmark_snapshot_refuses_resume(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ):
        pass
    changed = dict(identity, benchmark_snapshot_manifest_sha256="d" * 64)
    with pytest.raises(ValueError, match="identity does not match"):
        _open_run(_matcher(policy, fields), database, changed, inventory, resume=True)


def test_changed_shard_order_refuses_resume_even_with_same_manifest(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ):
        pass
    with pytest.raises(ValueError, match="ordered shard inventory"):
        _open_run(
            _matcher(policy, fields),
            database,
            identity,
            list(reversed(inventory)),
            resume=True,
        )


def test_manual_progress_sidecar_cannot_skip_uncommitted_records(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=2)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    matcher = _matcher(policy, fields)
    with _open_run(matcher, database, identity, inventory, resume=False) as run:
        for index, document in enumerate(
            _remaining_documents(root, inventory, manifest_identity, run)
        ):
            run.add_document(document)
            if index == 0:
                break
    edited_progress = tmp_path / "run.json"
    edited_progress.write_text(
        json.dumps({"next_shard_index": len(inventory), "next_row_ordinal": 0}),
        encoding="utf-8",
    )
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=True
    ) as run:
        assert run.resume_position == {
            "shard_index": 0,
            "row_ordinal": 0,
            "documents_seen": 0,
        }


@pytest.mark.parametrize(
    "change",
    [
        {"matcher_policy_sha256": "e" * 64},
        {"matcher_implementation_version": "unknown-matcher-version"},
    ],
)
def test_incompatible_policy_or_matcher_version_refuses_resume(tmp_path, change):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ):
        pass
    with pytest.raises(ValueError):
        _open_run(
            _matcher(policy, fields),
            database,
            dict(identity, **change),
            inventory,
            resume=True,
        )


@pytest.mark.parametrize("corruption", ["missing", "truncated"])
def test_missing_or_corrupted_checkpoint_refuses_resume(tmp_path, corruption):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ):
        pass
    if corruption == "missing":
        database.unlink()
    else:
        database.write_bytes(b"not a sqlite database")
    expected_error = (
        FileNotFoundError if corruption == "missing" else sqlite3.DatabaseError
    )
    with pytest.raises(expected_error):
        _open_run(_matcher(policy, fields), database, identity, inventory, resume=True)


def test_completed_checkpoint_can_be_reopened_without_duplicate_final_hits(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=False
    ) as run:
        run.scan(_remaining_documents(root, inventory, manifest_identity, run))
        original = [asdict(item) for item in run.iter_results()]
        original_count = run.connection.execute(
            "SELECT count(*) FROM final_hits"
        ).fetchone()[0]
    with _open_run(
        _matcher(policy, fields), database, identity, inventory, resume=True
    ) as run:
        assert run.checkpoint_status == "FINALIZED"
        assert run.finish().documents_seen == 9
        assert [asdict(item) for item in run.iter_results()] == original
        assert (
            run.connection.execute("SELECT count(*) FROM final_hits").fetchone()[0]
            == original_count
        )


def test_duplicate_and_out_of_order_input_rows_are_rejected(tmp_path):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy(commit_every_documents=2)
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    matcher = _matcher(policy, fields)
    first = next(_corpus_documents(root, inventory, manifest_identity))
    second = next(
        document
        for document in _corpus_documents(root, inventory, manifest_identity)
        if document.source_shard_index == 0 and document.source_row_ordinal == 1
    )
    with _open_run(matcher, database, identity, inventory, resume=False) as run:
        run.add_document(first)
        with pytest.raises(ValueError, match="Duplicate corpus record ID"):
            run.add_document(replace(second, doc_id=first.doc_id))

    out_of_order_db = tmp_path / "out-of-order.sqlite3"
    with _open_run(
        _matcher(policy, fields), out_of_order_db, identity, inventory, resume=False
    ) as run:
        with pytest.raises(ValueError, match="Out-of-order or unexpected source row"):
            run.add_document(second)


def test_manifest_checksums_and_atomic_publication_recover_after_rename_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "corpus"
    inventory, manifest_identity, fields = _make_corpus(root)
    policy = CandidatePolicy()
    identity = _identity(inventory, manifest_identity, policy)
    database = tmp_path / "checkpoint.sqlite3"
    output = tmp_path / "output"
    output.mkdir()
    completed = _finish_from_checkpoint(
        root,
        inventory,
        manifest_identity,
        _matcher(policy, fields),
        database,
        identity,
        output,
        resume=False,
    )
    outputs = {
        name: {
            "bytes": (output / name).stat().st_size,
            "sha256": sha256_file(output / name),
        }
        for name in completed["artifact_sha256"]
    }
    manifest = {
        "scan_version": SCAN_VERSION,
        "status": "BD3_SCAN_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "run_id": "artifact-verification-test",
        "checkpoint_identity_sha256": "f" * 64,
        "candidate_hit_rows": len(completed["results"]),
        "candidate_anchor_evidence_rows": len(completed["evidence"]),
        "outputs": outputs,
    }
    (output / "manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    (output / "manifest.sha256").write_text(
        sha256_file(output / "manifest.json") + "\n", encoding="ascii"
    )
    (output / "INCOMPLETE.json").write_text(
        canonical_json(
            {
                "status": "INCOMPLETE",
                "run_id": manifest["run_id"],
                "checkpoint_identity_sha256": manifest["checkpoint_identity_sha256"],
            }
        ),
        encoding="utf-8",
    )
    check = _verify_output_records(output, manifest, allow_incomplete_marker=True)
    assert check["candidate_hit_rows"] == len(completed["results"])
    assert check["candidate_anchor_evidence_rows"] == len(completed["evidence"])
    final_output = tmp_path / "published-output"
    replace = production_module.os.replace

    def fail_atomic_rename(source, destination):
        if Path(source) == output and Path(destination) == final_output:
            raise OSError("injected crash before atomic directory publication")
        return replace(source, destination)

    monkeypatch.setattr(production_module.os, "replace", fail_atomic_rename)
    with pytest.raises(OSError, match="atomic directory publication"):
        _publish_prepared_stage(
            output,
            final_output,
            run_id=manifest["run_id"],
            checkpoint_identity_sha256=manifest["checkpoint_identity_sha256"],
        )
    assert output.is_dir()
    assert not (output / "INCOMPLETE.json").exists()
    assert not final_output.exists()

    monkeypatch.setattr(production_module.os, "replace", replace)
    _publish_prepared_stage(
        output,
        final_output,
        run_id=manifest["run_id"],
        checkpoint_identity_sha256=manifest["checkpoint_identity_sha256"],
    )
    assert not output.exists()
    assert _verify_output_records(final_output, manifest)["candidate_hit_rows"] == len(
        completed["results"]
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _publish_prepared_stage(
            output,
            final_output,
            run_id=manifest["run_id"],
            checkpoint_identity_sha256=manifest["checkpoint_identity_sha256"],
        )

    hit_path = final_output / "candidate_hits.parquet"
    hit_size = hit_path.stat().st_size
    hit_path.write_bytes(b"x" * hit_size)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_output_records(final_output, manifest)
