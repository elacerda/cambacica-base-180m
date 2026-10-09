from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import tracemalloc

import pyarrow.parquet as pq
import pytest

from cambacica.corpus.decontamination.fixtures import (
    DEV_PASSAGE,
    synthetic_cases,
    synthetic_fields,
)
from cambacica.corpus.decontamination.matcher import (
    BenchmarkMatcher,
    CandidatePolicy,
    CorpusDocument,
    MatchField,
    _anchor_digest,
    normalize_match_text,
    tokenize_with_offsets,
)
from cambacica.corpus.decontamination import snapshot as snapshot_module
from cambacica.corpus.decontamination.production import (
    ANCHOR_EVIDENCE_SCHEMA,
    _write_anchor_evidence,
)


def _field(
    example_id: str,
    role: str,
    text: str,
    *,
    matchable: bool = True,
) -> MatchField:
    return MatchField(
        benchmark_name="test_benchmark",
        example_id=example_id,
        source_row_id=example_id,
        source_file_sha256=hashlib.sha256(b"test-source").hexdigest(),
        source_category=None,
        field_id=f"{example_id}:{role}",
        field_role=role,
        original_text=text,
        matchable=matchable,
    )


def _scan(fields, docs, tmp_path, policy=None):
    matcher = BenchmarkMatcher(fields, policy)
    with matcher.start_run(scratch_dir=tmp_path) as run:
        accounting = run.scan(iter(docs))
        results = list(run.iter_results())
    return accounting, results


def test_match_normalization_preserves_diacritics_and_source_offsets():
    original = "Prefixo Cafe\u0301 — ÁRVORE, Straße!"
    tokens = tokenize_with_offsets(original)

    assert [token.text for token in tokens] == ["prefixo", "café", "árvore", "strasse"]
    assert original[tokens[1].start : tokens[1].end] == "Cafe\u0301"
    assert normalize_match_text("Cafe\u0301") == "café"
    assert normalize_match_text("café") != normalize_match_text("cafe")


def test_blank_calame_target_is_preserved_without_complete_item_match(tmp_path):
    row = {
        "id": 718,
        "sentence": "Contexto integral do exemplo.",
        "last_word": "      ",
    }
    aggregate = tmp_path / "calamept_all.jsonl"
    generated = tmp_path / "calamept_gen_only.jsonl"
    handwritten = tmp_path / "calamept_handwritten_only.jsonl"
    aggregate.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    generated.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    handwritten.write_text("", encoding="utf-8")

    _registry, fields, counts = snapshot_module._calame_records(
        tmp_path, {"calamept_all.jsonl": hashlib.sha256(b"source").hexdigest()}
    )
    by_role = {field["field_role"]: field for field in fields}

    assert counts == {"all": 1, "generated": 1}
    assert by_role["context"]["matchable"] is True
    assert by_role["target_word"]["original_text"] == "      "
    assert by_role["target_word"]["matchable"] is False
    assert by_role["complete_item"]["matchable"] is False


def test_exact_question_and_span_offsets_with_unicode_punctuation(tmp_path):
    question = "Qual cidade recebeu a exposição científica no litoral?"
    text = f"Arquivo de campo — “{question.upper()}!!!” — registro preservado."
    fields = [_field("unicode-1", "question", question)]
    _accounting, results = _scan(
        fields,
        [CorpusDocument("record-1", text, source_shard="shard.parquet")],
        tmp_path,
    )

    assert len(results) == 1
    hit = results[0]
    assert hit.decision_rule == "exact_distinctive_question"
    assert (
        text[hit.corpus_char_start : hit.corpus_char_end].casefold()
        == question.rstrip("?").casefold()
    )
    assert hit.benchmark_char_start == 0
    assert hit.benchmark_char_end == len(question.rstrip("?"))


def test_exact_complete_item_is_reported_without_approximate_anchors(tmp_path):
    complete_item = (
        "Passagem: aves marinhas retornaram à ilha depois da tempestade.\n"
        "Pergunta: Qual grupo voltou primeiro?\nA. Andorinhas costeiras"
    )
    fields = [
        _field("complete-1", "complete_item", complete_item),
        _field("complete-1", "passage", complete_item),
    ]
    documents = [
        CorpusDocument(f"complete-copy-{index}", f"Arquivo. {complete_item} Fim.")
        for index in range(3)
    ]

    _accounting, results = _scan(
        fields, documents, tmp_path, CandidatePolicy(max_ngram_df=1)
    )

    complete_hits = [hit for hit in results if hit.field_role == "complete_item"]
    assert len(complete_hits) == 3
    assert all(hit.decision_rule == "exact_complete_item" for hit in complete_hits)
    assert all(hit.exact_match for hit in complete_hits)
    assert all(
        hit.contiguous_tokens == len(tokenize_with_offsets(complete_item))
        for hit in complete_hits
    )


def test_accents_are_not_stripped_for_exact_matching(tmp_path):
    fields = [_field("accent-1", "question", "O café mantém aroma na região serrana?")]
    docs = [CorpusDocument("record-1", "O cafe mantem aroma na regiao serrana?")]
    _accounting, results = _scan(fields, docs, tmp_path)
    assert results == []


def test_exact_thirteen_plus_token_question_is_distinctive_evidence(tmp_path):
    question = (
        "Qual estação costeira registrava a altura das marés na ilha de Pedra Clara "
        "durante o inverno frio?"
    )
    assert len(tokenize_with_offsets(question)) >= 13
    field = _field("question-13", "question", question)
    _accounting, results = _scan(
        [field], [CorpusDocument("question-13-copy", f"Arquivo: {question}")], tmp_path
    )

    exact_hit = next(
        result
        for result in results
        if result.decision_rule == "exact_distinctive_question"
    )
    assert exact_hit.matched_anchor_document_frequency_min is None


def test_repeated_benchmark_question_text_is_not_distinctive_by_itself(tmp_path):
    question = "Qual nome identifica o farol antigo?"
    fields = [
        _field(f"duplicate-question-{index}", "question", question)
        for index in range(3)
    ]
    _accounting, results = _scan(
        fields, [CorpusDocument("generic-question", question)], tmp_path
    )

    assert results == []


def test_answer_alone_is_not_a_candidate_but_question_and_answer_are(tmp_path):
    question = "Qual instrumento registrava a altura da água na ilha de Pedra Clara?"
    answer = "Um observatório de marés instalado pela engenheira Lídia Seranduva."
    fields = [
        _field("qa-1", "question", question),
        _field("qa-1", "correct_answer", answer, matchable=False),
        _field("qa-1", "question_plus_answer", f"{question} {answer}"),
    ]
    docs = [
        CorpusDocument("answer-only", "Resposta isolada: Nacarim."),
        CorpusDocument("answer-alone", answer),
        CorpusDocument("qa", f"{question} Resposta: {answer}"),
    ]
    _accounting, results = _scan(fields, docs, tmp_path)
    by_doc = {}
    for result in results:
        by_doc.setdefault(result.doc_id, []).append(result)
    assert "answer-only" not in by_doc
    assert "answer-alone" not in by_doc
    assert any(
        result.decision_rule == "exact_question_plus_answer" for result in by_doc["qa"]
    )


def test_fixture_suite_catches_positives_and_rejects_negative_controls(tmp_path):
    cases = synthetic_cases()
    fields = synthetic_fields()
    _accounting, results = _scan(
        fields,
        [case.document for case in cases],
        tmp_path,
        CandidatePolicy(),
    )
    detected = {result.doc_id for result in results}
    mismatches = [
        case.case_id
        for case in cases
        if (case.document.doc_id in detected) != case.expected_match
    ]
    assert mismatches == []
    assert any(
        result.decision_rule == "distinctive_anchor_coverage" for result in results
    )
    assert any(result.decision_rule == "contiguous_token_run" for result in results)


def test_contiguous_partial_passage_embedded_in_a_longer_document(tmp_path):
    passage_tokens = [token.text for token in tokenize_with_offsets(DEV_PASSAGE)]
    partial = " ".join(passage_tokens[:50])
    field = _field("partial-1", "passage", DEV_PASSAGE)
    doc = CorpusDocument("embedded", f"Header text. {partial} Footer text.")
    matcher = BenchmarkMatcher([field])
    with matcher.start_run(scratch_dir=tmp_path) as run:
        _accounting = run.scan([doc])
        results = list(run.iter_results())
        anchor_spans = list(run.iter_anchor_evidence())
    hit = next(result for result in results if result.doc_id == "embedded")
    assert hit.decision_rule == "contiguous_token_run"
    assert hit.contiguous_tokens >= 50
    assert hit.matched_anchor_document_frequency_min == 1
    assert hit.matched_anchor_document_frequency_max == 1
    assert anchor_spans
    for span in anchor_spans:
        corpus_excerpt = doc.text[span.corpus_char_start : span.corpus_char_end]
        benchmark_excerpt = DEV_PASSAGE[
            span.benchmark_char_start : span.benchmark_char_end
        ]
        assert (
            " ".join(token.text for token in tokenize_with_offsets(corpus_excerpt))
            == span.anchor_text
        )
        assert (
            " ".join(token.text for token in tokenize_with_offsets(benchmark_excerpt))
            == span.anchor_text
        )
        assert span.anchor_document_frequency == 1


def test_empty_anchor_evidence_parquet_has_stable_schema(tmp_path):
    output = tmp_path / "candidate_anchor_evidence.parquet"
    assert _write_anchor_evidence(output, iter(())) == 0
    parquet = pq.ParquetFile(output)
    assert parquet.schema_arrow.equals(ANCHOR_EVIDENCE_SCHEMA)
    assert parquet.metadata.num_rows == 0


def test_exact_field_match_crosses_token_chunk_boundary_with_source_offsets(tmp_path):
    prefix = " ".join(f"antecedente{index}" for index in range(180))
    text = f"{prefix} {DEV_PASSAGE} rodape final"
    matcher = BenchmarkMatcher(
        [_field("chunk-boundary", "passage", DEV_PASSAGE)],
        CandidatePolicy(document_chunk_tokens=256),
    )
    document = CorpusDocument("chunked-record", text)

    with matcher.start_run(scratch_dir=tmp_path) as run:
        accounting = run.scan(iter([document]))
        results = list(run.iter_results())

    assert accounting.documents_seen == 1
    hit = next(item for item in results if item.decision_rule == "exact_passage")
    assert hit.decision_rule == "exact_passage"
    assert hit.corpus_token_start == 180
    assert text[hit.corpus_char_start : hit.corpus_char_end] == DEV_PASSAGE.rstrip(".")


def test_large_document_python_memory_scales_with_token_window(tmp_path):
    text = " ".join(f"palavra{index}" for index in range(30_000))
    matcher = BenchmarkMatcher(
        synthetic_fields(), CandidatePolicy(document_chunk_tokens=512)
    )
    tracemalloc.start()
    try:
        with matcher.start_run(scratch_dir=tmp_path) as run:
            accounting = run.scan(iter([CorpusDocument("large-no-hit", text)]))
            _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert accounting.documents_seen == 1
    assert accounting.candidate_fields == 0
    assert peak < 8 * 1024 * 1024


def test_short_exact_question_uses_document_frequency(tmp_path):
    question = "Qual nome identifica o farol antigo?"
    field = _field("short-1", "question", question)
    policy = CandidatePolicy(max_exact_question_document_frequency=2)
    docs = [
        CorpusDocument(f"question-copy-{index}", question, source_shard=f"part-{index}")
        for index in range(3)
    ]
    _accounting, results = _scan([field], docs, tmp_path, policy)
    assert results == []


def test_distinct_document_frequency_ignores_repetition_and_discards_promoted_anchor(
    tmp_path,
):
    anchor = "porto sereno relata maresia sobre pedras muito antigas".split()
    anchor = (anchor + [f"termo{index}" for index in range(13)])[:13]
    tail = [f"conteudo{index}" for index in range(20)]
    field = _field("frequency-1", "passage", " ".join(anchor + tail))
    policy = CandidatePolicy(max_ngram_df=2)
    repeated = " ".join(anchor * 25)
    docs = [
        CorpusDocument("copy-a", repeated, source_shard="a.parquet"),
        CorpusDocument(
            "copy-b", "prefixo " + " ".join(anchor), source_shard="b.parquet"
        ),
        CorpusDocument(
            "copy-c", "prefixo " + " ".join(anchor), source_shard="c.parquet"
        ),
    ]
    matcher = BenchmarkMatcher([field], policy)
    with matcher.start_run(scratch_dir=tmp_path, keep_scratch=True) as run:
        run.scan(iter(docs))
        anchor_id = _anchor_digest(tuple(anchor))
        assert run.anchor_frequencies()[anchor_id] == 3
        assert (
            run.connection.execute(
                "SELECT count(*) FROM anchor_evidence WHERE anchor=?", (anchor_id,)
            ).fetchone()[0]
            == 0
        )
        assert list(run.iter_results()) == []
        sqlite_path = run.db_path
    assert sqlite_path.exists()
    sqlite_path.unlink()


def test_repeated_anchor_in_one_document_counts_once(tmp_path):
    anchor = "porto sereno relata maresia sobre pedras muito antigas".split()
    anchor = (anchor + [f"termo{index}" for index in range(13)])[:13]
    field = _field("frequency-one", "passage", " ".join(anchor + ["cauda"] * 20))
    policy = CandidatePolicy(max_ngram_df=2)
    doc = CorpusDocument("one-document", " ".join(anchor * 40))
    matcher = BenchmarkMatcher([field], policy)
    with matcher.start_run(scratch_dir=tmp_path) as run:
        run.scan(iter([doc]))
        anchor_id = _anchor_digest(tuple(anchor))
        assert run.anchor_frequencies()[anchor_id] == 1


def test_scan_order_is_stable_and_duplicate_record_ids_fail(tmp_path):
    cases = [case for case in synthetic_cases() if case.expected_match]
    fields = synthetic_fields()
    _accounting_a, results_a = _scan(
        fields, [case.document for case in cases], tmp_path
    )
    _accounting_b, results_b = _scan(
        fields, [case.document for case in cases], tmp_path
    )
    assert [asdict(item) for item in results_a] == [asdict(item) for item in results_b]
    assert [
        (item.doc_id, item.example_id, item.field_id, item.decision_rule)
        for item in results_a
    ] == sorted(
        (item.doc_id, item.example_id, item.field_id, item.decision_rule)
        for item in results_a
    )
    matcher = BenchmarkMatcher(fields)
    with matcher.start_run(scratch_dir=tmp_path) as run:
        with pytest.raises(ValueError, match="Duplicate corpus record ID"):
            run.scan(
                iter(
                    [
                        CorpusDocument("duplicate", "primeiro texto"),
                        CorpusDocument("duplicate", "segundo texto"),
                    ]
                )
            )


def test_empty_documents_keep_python_memory_bounded_by_spill(tmp_path):
    matcher = BenchmarkMatcher(synthetic_fields())
    tracemalloc.start()
    with matcher.start_run(scratch_dir=tmp_path) as run:
        accounting = run.scan(
            CorpusDocument(
                f"no-hit-{index:05d}", "Texto sem correspondência específica."
            )
            for index in range(500)
        )
        current, peak = tracemalloc.get_traced_memory()
        scratch_bytes = run.scratch_bytes
    tracemalloc.stop()
    assert accounting.documents_seen == 500
    assert accounting.candidate_fields == 0
    assert peak < 8 * 1024 * 1024
    assert current < 8 * 1024 * 1024
    assert scratch_bytes < 4 * 1024 * 1024


def test_interrupted_scan_cleans_temporary_sqlite(tmp_path):
    matcher = BenchmarkMatcher(synthetic_fields())

    def broken_documents():
        yield CorpusDocument("first", "texto sem correspondência")
        raise RuntimeError("interrupted fixture")

    with pytest.raises(RuntimeError, match="interrupted fixture"):
        with matcher.start_run(scratch_dir=tmp_path) as run:
            run.scan(broken_documents())
    assert list(tmp_path.glob("cambacica-bd2-match-*.sqlite3")) == []


def test_high_frequency_scratch_database_is_removed_after_success(tmp_path):
    field = _field(
        "cleanup", "question", "Qual estação recebeu as medições da corrente costeira?"
    )
    matcher = BenchmarkMatcher([field])
    with matcher.start_run(scratch_dir=tmp_path) as run:
        run.scan(iter([CorpusDocument("one", field.original_text)]))
        assert run.scratch_bytes > 0
    assert list(tmp_path.glob("cambacica-bd2-match-*.sqlite3")) == []


def test_snapshot_download_failure_removes_incomplete_stage(tmp_path, monkeypatch):
    class UnavailablePinnedRepo:
        def dataset_info(self, *_args, **_kwargs):
            raise RuntimeError("pinned revision unavailable")

    monkeypatch.setattr(snapshot_module, "HfApi", UnavailablePinnedRepo)
    output = tmp_path / "snapshot"

    with pytest.raises(RuntimeError, match="pinned revision unavailable"):
        snapshot_module.build_snapshot(output)

    assert not output.exists()
    assert list(tmp_path.glob(".snapshot.*.incomplete")) == []
