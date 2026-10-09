#!/usr/bin/env python3
"""Build a blinded 40-case matcher review panel from the pinned BD2 snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cambacica.corpus.decontamination.matcher import (  # noqa: E402
    BenchmarkMatcher,
    CandidatePolicy,
    CorpusDocument,
    MatchField,
    fields_from_snapshot,
    tokenize_with_offsets,
)
from cambacica.corpus.decontamination.production import (  # noqa: E402
    PINNED_EXACT_MANIFEST_SHA256,
)
from cambacica.corpus.decontamination.snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    canonical_json,
    sha256_file,
    verify_snapshot,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2-preflight"
)
CALAME_REVISION = "353671bc95cc3d94d488201f67d41b640eb80c55"
BELEBELE_REVISION = "d4c91dedc9de484dbea7b7d940f898f59fd135e9"
PANEL_SEED = "c1-bd2.5-independent-panel-v1"

RESULT_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("prediction", pa.string(), nullable=False),
        pa.field("detected", pa.bool_(), nullable=False),
        pa.field("matched_example_ids_json", pa.string(), nullable=False),
        pa.field("matched_fields_json", pa.string(), nullable=False),
        pa.field("decision_rules_json", pa.string(), nullable=False),
        pa.field("matched_spans_json", pa.string(), nullable=False),
        pa.field("anchor_evidence_json", pa.string(), nullable=False),
        pa.field("document_text_sha256", pa.string(), nullable=False),
        pa.field("document_characters", pa.int64(), nullable=False),
    ]
)


def _rank(example_id: str) -> str:
    return hashlib.sha256(f"{PANEL_SEED}\0{example_id}".encode("utf-8")).hexdigest()


def _excerpt(text: str, max_chars: int = 1600) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[excerto truncado]"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_snapshot(
    snapshot_dir: Path,
) -> tuple[
    list[MatchField], dict[str, dict[str, Any]], dict[str, dict[str, MatchField]]
]:
    fields = fields_from_snapshot(snapshot_dir / "match_fields.parquet")
    registry = pq.read_table(snapshot_dir / "benchmark_registry.parquet").to_pylist()
    rows_by_example = {row["example_id"]: row for row in registry}
    fields_by_example: dict[str, dict[str, MatchField]] = {}
    for field in fields:
        fields_by_example.setdefault(field.example_id, {})[field.field_role] = field
    return fields, rows_by_example, fields_by_example


def _used_in_bd2_calibration() -> set[str]:
    path = Path(
        "/mnt/data/cambacica-base-180m/decontamination/benchmark-v1/bd2/calibration/fixture_results.parquet"
    )
    result: set[str] = set()
    if path.exists():
        table = pq.read_table(path, columns=["expected_example_id"])
        for value in table.column(0).to_pylist():
            if value and len(value) == 64:
                result.add(value)
    return result


def _select_examples(
    benchmark: str,
    fields_by_example: dict[str, dict[str, MatchField]],
    rows_by_example: dict[str, dict[str, Any]],
    used_ids: set[str],
    used_passage_ids: set[str],
    *,
    role: str,
    predicate,
    category_counts: dict[str, int] | None = None,
) -> tuple[str, dict[str, MatchField], dict[str, Any]]:
    candidates = []
    for example_id, role_fields in fields_by_example.items():
        registry = rows_by_example.get(example_id)
        if registry is None or registry["benchmark_name"] != benchmark:
            continue
        if example_id in used_ids or role not in role_fields:
            continue
        if not role_fields[role].matchable:
            continue
        if not predicate(role_fields, registry):
            continue
        passage_id = str(registry["source_row_id"]).split("#q", 1)[0]
        if benchmark == "belebele_por_latn" and passage_id in used_passage_ids:
            continue
        category = registry.get("source_category")
        category_priority = category_counts.get(category, 0) if category_counts else 0
        candidates.append(
            (category_priority, _rank(example_id), example_id, role_fields, registry)
        )
    if not candidates:
        raise ValueError(f"No unused example satisfies {benchmark}/{role}")
    _, _, example_id, role_fields, registry = min(candidates)
    used_ids.add(example_id)
    if benchmark == "belebele_por_latn":
        used_passage_ids.add(str(registry["source_row_id"]).split("#q", 1)[0])
    if category_counts is not None:
        category_counts[registry["source_category"]] = (
            category_counts.get(registry["source_category"], 0) + 1
        )
    return example_id, role_fields, registry


def _modify_case(text: str) -> str:
    tokens = [token.text for token in tokenize_with_offsets(text)]
    if len(tokens) < 8:
        raise ValueError("Insertion/deletion control requires at least eight tokens")
    insertion = max(2, len(tokens) // 3)
    deletion = min(len(tokens) - 2, (2 * len(tokens)) // 3)
    tokens.insert(insertion, "insercaocontrolada")
    if deletion >= insertion:
        deletion += 1
    del tokens[deletion]
    return " ".join(tokens)


def _case_punctuation(text: str) -> str:
    return f"“{text.swapcase()}”!!!"


def _case_id(benchmark: str, example_id: str, control: str) -> str:
    key = f"{PANEL_SEED}\0{benchmark}\0{example_id}\0{control}"
    return "C1BD25-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _build_case(
    benchmark: str,
    example_id: str,
    role_fields: dict[str, MatchField],
    registry: dict[str, Any],
    control: str,
    expected_class: str,
    document_text: str,
    matching_excerpt: str,
    relationship: str,
    provenance: str,
) -> dict[str, Any]:
    original_role = "context" if benchmark == "calame_pt" else "passage"
    if control in {"short_distinctive_item", "short_question"}:
        original_role = "context" if benchmark == "calame_pt" else "question"
    original_field = role_fields.get(original_role) or role_fields.get("complete_item")
    original_text = original_field.original_text if original_field else ""
    document_id = _case_id(benchmark, example_id, control)
    document = CorpusDocument(
        doc_id=document_id,
        text=document_text,
        source_shard="synthetic-preflight/" + control,
        source="synthetic_controlled_document",
        source_row_ordinal=None,
        input_manifest_sha256=None,
    )
    return {
        "case_id": document_id,
        "benchmark_name": benchmark,
        "benchmark_revision": CALAME_REVISION
        if benchmark == "calame_pt"
        else BELEBELE_REVISION,
        "example_id": example_id,
        "source_row_id": registry["source_row_id"],
        "source_category": registry.get("source_category"),
        "control_kind": control,
        "expected_class": expected_class,
        "expected_relationship": relationship,
        "known_construction_provenance": provenance,
        "original_excerpt": _excerpt(original_text),
        "matching_excerpt": _excerpt(matching_excerpt),
        "document": document,
        "document_sha256": hashlib.sha256(document_text.encode("utf-8")).hexdigest(),
    }


def _positive_cases(
    fields_by_example: dict[str, dict[str, MatchField]],
    rows_by_example: dict[str, dict[str, Any]],
    used_ids: set[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    category_counts = {"handwritten": 0, "generated": 0}
    calame_controls = [
        ("exact_complete_item", "complete_item", lambda roles, _r: True),
        ("exact_context", "context", lambda roles, _r: True),
        ("embedded_complete_item", "complete_item", lambda roles, _r: True),
        ("case_punctuation", "context", lambda roles, _r: True),
        (
            "insertions_deletions",
            "context",
            lambda roles, _r: (
                roles["context"].original_text
                and len(tokenize_with_offsets(roles["context"].original_text)) >= 100
            ),
        ),
        (
            "partial_contiguous_overlap",
            "context",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["context"].original_text)) >= 70
            ),
        ),
        (
            "long_context_embedded",
            "context",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["context"].original_text)) >= 120
            ),
        ),
        (
            "near_copy_punctuation",
            "context",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["context"].original_text)) >= 70
            ),
        ),
        (
            "short_distinctive_item",
            "context",
            lambda roles, _r: (
                13 <= len(tokenize_with_offsets(roles["context"].original_text)) <= 50
            ),
        ),
        ("context_plus_target", "context", lambda roles, _r: "target_word" in roles),
    ]
    for control, role, predicate in calame_controls:
        example_id, role_fields, registry = _select_examples(
            "calame_pt",
            fields_by_example,
            rows_by_example,
            used_ids,
            set(),
            role=role,
            predicate=lambda roles, row: (
                predicate(roles, row)
                and bool(
                    tokenize_with_offsets(
                        roles.get("target_word", _null_field()).original_text
                    )
                )
            ),
            category_counts=category_counts,
        )
        context = role_fields["context"].original_text
        complete = role_fields["complete_item"].original_text
        if control == "exact_complete_item":
            matching = document_text = complete
        elif control == "exact_context":
            matching = document_text = context
        elif control == "embedded_complete_item":
            matching = complete
            document_text = f"Cabeçalho do arquivo. {complete} Fim do registro."
        elif control in {"case_punctuation", "near_copy_punctuation"}:
            matching = _case_punctuation(context)
            document_text = f"Trecho preservado: {matching} Encerramento."
        elif control == "insertions_deletions":
            matching = _modify_case(context)
            document_text = f"Versão transcrita: {matching} Fim."
        elif control == "partial_contiguous_overlap":
            tokens = [token.text for token in tokenize_with_offsets(context)]
            matching = " ".join(tokens[5:65])
            document_text = f"Trecho parcial: {matching} Fim do trecho."
        elif control == "long_context_embedded":
            matching = context
            document_text = f"Relatório independente. {context} Anotação final."
        elif control == "short_distinctive_item":
            matching = context
            document_text = f"Recorte exato: {context} Fim."
        elif control == "context_plus_target":
            matching = context + " " + role_fields["target_word"].original_text
            document_text = f"Ficha completa: {matching} Arquivo encerrado."
        else:
            raise ValueError(f"Unhandled CALAME control: {control}")
        cases.append(
            _build_case(
                "calame_pt",
                example_id,
                role_fields,
                registry,
                control,
                "positive",
                document_text,
                matching,
                "Controlled insertion of the pinned CALAME material into a synthetic document.",
                "The benchmark excerpt is copied from the pinned snapshot; wrappers and edits are generated deterministically for this panel.",
            )
        )

    used_passage_ids: set[str] = set()
    belebele_controls = [
        ("exact_complete_item", "complete_item", lambda roles, _r: True),
        ("exact_passage", "passage", lambda roles, _r: True),
        ("question_plus_answer", "question_plus_answer", lambda roles, _r: True),
        (
            "long_passage_embedded",
            "passage",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["passage"].original_text)) >= 150
            ),
        ),
        ("case_punctuation", "passage", lambda roles, _r: True),
        (
            "insertions_deletions",
            "passage",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["passage"].original_text)) >= 100
            ),
        ),
        (
            "partial_contiguous_overlap",
            "passage",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["passage"].original_text)) >= 70
            ),
        ),
        (
            "short_question",
            "question",
            lambda roles, _r: (
                6 <= len(tokenize_with_offsets(roles["question"].original_text)) <= 50
            ),
        ),
        (
            "near_copy_punctuation",
            "passage",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["passage"].original_text)) >= 70
            ),
        ),
        (
            "exact_question",
            "question",
            lambda roles, _r: (
                len(tokenize_with_offsets(roles["question"].original_text)) >= 6
            ),
        ),
    ]
    for control, role, predicate in belebele_controls:
        example_id, role_fields, registry = _select_examples(
            "belebele_por_latn",
            fields_by_example,
            rows_by_example,
            used_ids,
            used_passage_ids,
            role=role,
            predicate=predicate,
        )
        if control == "exact_complete_item":
            matching = document_text = role_fields["complete_item"].original_text
        elif control == "exact_passage":
            matching = document_text = role_fields["passage"].original_text
        elif control == "question_plus_answer":
            matching = document_text = role_fields["question_plus_answer"].original_text
        elif control == "long_passage_embedded":
            matching = role_fields["passage"].original_text
            document_text = f"Relatório de leitura. {matching} Conclusão do relatório."
        elif control == "case_punctuation":
            matching = _case_punctuation(role_fields["passage"].original_text)
            document_text = f"Transcrição revisada: {matching} Fim."
        elif control == "insertions_deletions":
            matching = _modify_case(role_fields["passage"].original_text)
            document_text = f"Cópia com pequenas alterações: {matching} Fim."
        elif control == "partial_contiguous_overlap":
            tokens = [
                token.text
                for token in tokenize_with_offsets(role_fields["passage"].original_text)
            ]
            matching = " ".join(tokens[8:68])
            document_text = f"Trecho extraído: {matching} Encerramento."
        elif control in {"short_question", "exact_question"}:
            matching = role_fields["question"].original_text
            document_text = f"Pergunta anotada: {matching} Fim."
        elif control == "near_copy_punctuation":
            matching = _case_punctuation(role_fields["passage"].original_text)
            document_text = f"Cópia pública. {matching} Registro concluído."
        else:
            raise ValueError(f"Unhandled Belebele control: {control}")
        cases.append(
            _build_case(
                "belebele_por_latn",
                example_id,
                role_fields,
                registry,
                control,
                "positive",
                document_text,
                matching,
                "Controlled insertion of the pinned Belebele passage or item into a synthetic document.",
                "The benchmark excerpt is copied from the pinned snapshot; wrappers and edits are generated deterministically for this panel.",
            )
        )
    return cases


def _null_field() -> MatchField:
    return MatchField("", "", "", "", None, "", "", "", False)


def _topic_hint(text: str) -> str:
    tokens = [token.text for token in tokenize_with_offsets(text)]
    return next((token for token in tokens if len(token) >= 7), "assunto")


def _scattered_fragments(text: str) -> tuple[str, str]:
    tokens = [token.text for token in tokenize_with_offsets(text)]
    if len(tokens) < 100:
        raise ValueError("Scattered-fragment negative requires at least 100 tokens")
    block = 20
    step = 24
    fragments = [
        tokens[start : start + block] for start in range(0, len(tokens) - block, step)
    ]
    if len(fragments) < 4:
        fragments = [tokens[:20], tokens[30:50], tokens[60:80], tokens[-20:]]
    filler = [f"conteudoindependente{index}" for index in range(1800)]
    document_parts: list[str] = []
    for index, fragment in enumerate(fragments):
        if index:
            document_parts.extend(filler)
        document_parts.extend(fragment)
    excerpt = " …[1,800 tokens não relacionados]… ".join(
        " ".join(fragment) for fragment in fragments[:4]
    )
    return " ".join(document_parts), excerpt


def _negative_document(
    control: str, role_fields: dict[str, MatchField]
) -> tuple[str, str, str]:
    if control == "answer_only":
        answer = role_fields.get("target_word") or role_fields.get("correct_answer")
        text = answer.original_text if answer else ""
        return (
            text,
            text,
            "Only the answer text is present; no benchmark question or passage is included.",
        )
    if control == "generic_portuguese":
        text = "A equipe organizou os documentos, conferiu as datas e registrou os resultados em uma planilha simples."
        return text, text, "Generic Portuguese prose with no benchmark field copied."
    if control == "multiple_choice_instructions":
        text = "Leia as instruções com atenção. Analise as alternativas e assinale somente uma resposta."
        return (
            text,
            text,
            "Common multiple-choice directions without benchmark-specific question or passage content.",
        )
    source = (
        role_fields.get("context")
        or role_fields.get("passage")
        or role_fields.get("question")
    )
    source_text = source.original_text if source else ""
    if control == "topical_different":
        hint = _topic_hint(source_text)
        text = f"Em outro estudo sobre {hint}, a equipe registrou uma observação diferente e publicou dados de outra época."
        return (
            text,
            text,
            "A manually written theme-adjacent statement uses one topic word but states different facts and wording.",
        )
    if control == "common_bibliography":
        text = (
            "Referências: Almeida, R. (2018). Estudos de linguagem. Editora Horizonte.\n"
            "Silva, M. (2020). Métodos de pesquisa e análise. Revista de Estudos.\n"
            "Costa, P. (2022). Introdução à documentação. Porto: Arquivo Aberto."
        )
        return (
            text,
            text,
            "Synthetic common-reference list with no benchmark passage or question copied.",
        )
    if control == "shared_public_quotation":
        text = "A ocasião faz o ladrão. Quem tudo quer, tudo perde."
        return (
            text,
            text,
            "Two short, widely circulated Portuguese proverbs are used as a shared-quotation control.",
        )
    if control == "distinct_entities_shared_template":
        text = (
            "A espécie fictícia Luminara azul foi observada no vale do Cedro e apresenta folhas estreitas. "
            "A espécie fictícia Nebulosa dourada foi registrada no planalto do Sul e apresenta folhas largas. "
            "Ambas as descrições seguem o mesmo formato, mas referem-se a entidades e fatos diferentes."
        )
        return (
            text,
            text,
            "Two distinct fictional entities share a repeated descriptive template.",
        )
    if control == "distant_fragments":
        text, excerpt = _scattered_fragments(source_text)
        return (
            text,
            excerpt,
            "Many real benchmark fragments are inserted into one synthetic document with long unrelated gaps; no localized aligned copy is constructed.",
        )
    if control == "repeated_boilerplate":
        repeated = "O presente documento foi recebido, revisado e incluído no arquivo correspondente."
        text = " ".join(f"Registro {index}: {repeated}" for index in range(120))
        return (
            text,
            repeated,
            "A generic document template is repeated at distant positions without benchmark-specific text.",
        )
    if control == "topical_different_second":
        hint = _topic_hint(source_text)
        text = f"O tema {hint} foi discutido em uma reunião diferente, com participantes e conclusões que não aparecem no item avaliado."
        return (
            text,
            text,
            "A second manually written topic-adjacent control describes different people, events and conclusions.",
        )
    raise ValueError(f"Unknown hard-negative control: {control}")


def _negative_cases(
    fields_by_example: dict[str, dict[str, MatchField]],
    rows_by_example: dict[str, dict[str, Any]],
    used_ids: set[str],
) -> list[dict[str, Any]]:
    controls = [
        "answer_only",
        "generic_portuguese",
        "multiple_choice_instructions",
        "topical_different",
        "common_bibliography",
        "shared_public_quotation",
        "distinct_entities_shared_template",
        "distant_fragments",
        "repeated_boilerplate",
        "topical_different_second",
    ]
    cases = []
    used_passage_ids: set[str] = set()
    for benchmark in ("calame_pt", "belebele_por_latn"):
        for control in controls:
            role = "context" if benchmark == "calame_pt" else "passage"
            if control == "answer_only":
                role = "context" if benchmark == "calame_pt" else "question"
            if control == "distant_fragments":
                role = "context" if benchmark == "calame_pt" else "passage"
            if control in {"answer_only", "distant_fragments"}:

                def predicate(roles, _row):
                    if control == "distant_fragments":
                        return (
                            len(tokenize_with_offsets(roles[role].original_text)) >= 100
                        )
                    return bool(
                        tokenize_with_offsets(
                            (
                                roles.get("target_word")
                                or roles.get("correct_answer")
                                or _null_field()
                            ).original_text
                        )
                    )
            else:

                def predicate(_roles, _row):
                    return True

            example_id, role_fields, registry = _select_examples(
                benchmark,
                fields_by_example,
                rows_by_example,
                used_ids,
                used_passage_ids,
                role=role,
                predicate=predicate,
            )
            document_text, matching_excerpt, provenance = _negative_document(
                control, role_fields
            )
            cases.append(
                _build_case(
                    benchmark,
                    example_id,
                    role_fields,
                    registry,
                    control,
                    "negative",
                    document_text,
                    matching_excerpt,
                    "No localized benchmark exposure is constructed; this is a synthetic hard-negative control.",
                    provenance,
                )
            )
    return cases


def _collect_predictions(
    fields: list[MatchField], cases: list[dict[str, Any]], scratch_dir: Path
) -> list[dict[str, Any]]:
    matcher = BenchmarkMatcher(fields, CandidatePolicy())
    by_doc: dict[str, list[Any]] = {}
    anchors_by_doc: dict[str, list[Any]] = {}
    with matcher.start_run(scratch_dir=scratch_dir) as run:
        run.scan(case["document"] for case in cases)
        for result in run.iter_results():
            by_doc.setdefault(result.doc_id, []).append(result)
        for evidence in run.iter_anchor_evidence():
            anchors_by_doc.setdefault(evidence.doc_id, []).append(evidence)
    rows = []
    for case in cases:
        case_hits = sorted(
            by_doc.get(case["case_id"], []),
            key=lambda item: (
                item.example_id,
                item.field_role,
                item.decision_rule,
                item.corpus_token_start,
            ),
        )
        case_anchors = sorted(
            anchors_by_doc.get(case["case_id"], []),
            key=lambda item: (
                item.example_id,
                item.field_role,
                item.benchmark_token_start,
                item.corpus_token_start,
            ),
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "benchmark_name": case["benchmark_name"],
                "example_id": case["example_id"],
                "prediction": "candidate" if case_hits else "no_candidate",
                "detected": bool(case_hits),
                "matched_example_ids_json": canonical_json(
                    sorted({item.example_id for item in case_hits})
                ),
                "matched_fields_json": canonical_json(
                    sorted({item.field_id for item in case_hits})
                ),
                "decision_rules_json": canonical_json(
                    sorted({item.decision_rule for item in case_hits})
                ),
                "matched_spans_json": canonical_json(
                    [
                        {
                            "example_id": item.example_id,
                            "field_id": item.field_id,
                            "field_role": item.field_role,
                            "decision_rule": item.decision_rule,
                            "exact_match": item.exact_match,
                            "matched_tokens": item.matched_tokens,
                            "contiguous_tokens": item.contiguous_tokens,
                            "distinctive_anchor_count": item.distinctive_anchor_count,
                            "distinctive_token_coverage": item.distinctive_token_coverage,
                            "anchor_df_min": item.matched_anchor_document_frequency_min,
                            "anchor_df_max": item.matched_anchor_document_frequency_max,
                            "corpus_token_start": item.corpus_token_start,
                            "corpus_token_end": item.corpus_token_end,
                            "corpus_char_start": item.corpus_char_start,
                            "corpus_char_end": item.corpus_char_end,
                            "benchmark_token_start": item.benchmark_token_start,
                            "benchmark_token_end": item.benchmark_token_end,
                            "benchmark_char_start": item.benchmark_char_start,
                            "benchmark_char_end": item.benchmark_char_end,
                        }
                        for item in case_hits
                    ]
                ),
                "anchor_evidence_json": canonical_json(
                    [
                        {
                            "anchor_sha256": item.anchor_sha256,
                            "anchor_text": item.anchor_text,
                            "anchor_document_frequency": item.anchor_document_frequency,
                            "corpus_token_start": item.corpus_token_start,
                            "corpus_token_end": item.corpus_token_end,
                            "corpus_char_start": item.corpus_char_start,
                            "corpus_char_end": item.corpus_char_end,
                            "benchmark_token_start": item.benchmark_token_start,
                            "benchmark_token_end": item.benchmark_token_end,
                            "benchmark_char_start": item.benchmark_char_start,
                            "benchmark_char_end": item.benchmark_char_end,
                        }
                        for item in case_anchors
                    ]
                ),
                "document_text_sha256": case["document_sha256"],
                "document_characters": len(case["document"].text),
            }
        )
    return rows


def _make_manifest(
    output_dir: Path,
    snapshot_check: dict[str, Any],
    cases: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {row["case_id"]: row["expected_class"] for row in cases}
    expected_example = {row["case_id"]: row["example_id"] for row in cases}
    correct_positive = sum(
        expected[row["case_id"]] == "positive"
        and expected_example[row["case_id"]]
        in json.loads(row["matched_example_ids_json"])
        for row in result_rows
    )
    clean_negative = sum(
        expected[row["case_id"]] == "negative" and not row["detected"]
        for row in result_rows
    )
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "preflight_manifest.json":
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "preflight_version": "c1-bd2.5-preflight-v1",
        "status": "PREFLIGHT_EVIDENCE_PREPARED",
        "scientist_signoff": "PENDING",
        "bd3_production_scan": "NOT_RUN_NOT_APPROVED",
        "inputs": {
            "post_d1_exact_manifest_sha256": PINNED_EXACT_MANIFEST_SHA256,
            "benchmark_snapshot_manifest_sha256": snapshot_check["manifest_sha256"],
            "calame_pt_revision": CALAME_REVISION,
            "belebele_revision": BELEBELE_REVISION,
            "bd2_fixture_manifest_sha256": "904fd5dbe8175b7b2c216af69e4a592a5cca20e9314ac21c694b10471e2ea28e",
        },
        "panel": {
            "case_count": len(cases),
            "calame_positive": sum(
                row["benchmark_name"] == "calame_pt"
                and row["expected_class"] == "positive"
                for row in cases
            ),
            "calame_negative": sum(
                row["benchmark_name"] == "calame_pt"
                and row["expected_class"] == "negative"
                for row in cases
            ),
            "belebele_positive": sum(
                row["benchmark_name"] == "belebele_por_latn"
                and row["expected_class"] == "positive"
                for row in cases
            ),
            "belebele_negative": sum(
                row["benchmark_name"] == "belebele_por_latn"
                and row["expected_class"] == "negative"
                for row in cases
            ),
            "positive_expected_class_controls_detected": correct_positive,
            "negative_expected_class_controls_without_candidate": clean_negative,
            "all_expected_classes_human_verified": False,
            "all_controls_synthetic": True,
            "held_out_from_bd2_fixture_example_ids": True,
            "calame_inventory_rows": 2076,
            "calame_evaluable_rows_metric_convention": 2075,
            "calame_whitespace_target_row_excluded_from_accuracy_denominator": 718,
        },
        "artifacts": artifacts,
    }
    return manifest


def build_panel(
    output_dir: Path, snapshot_dir: Path, scratch_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    for name in (
        "review_panel_blind.csv",
        "review_panel_answer_key.csv",
        "review_panel_provisional_notes.md",
        "fixture_results.parquet",
    ):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite preflight artifact: {name}")
    snapshot_check = verify_snapshot(snapshot_dir)
    fields, rows_by_example, fields_by_example = _load_snapshot(snapshot_dir)
    used_ids = _used_in_bd2_calibration()
    positive_cases = _positive_cases(fields_by_example, rows_by_example, used_ids)
    negative_cases = _negative_cases(fields_by_example, rows_by_example, used_ids)
    cases = positive_cases + negative_cases
    if len(cases) != 40:
        raise ValueError(
            f"Independent panel must contain exactly 40 cases, found {len(cases)}"
        )
    counts = {
        (benchmark, expected): sum(
            row["benchmark_name"] == benchmark and row["expected_class"] == expected
            for row in cases
        )
        for benchmark in ("calame_pt", "belebele_por_latn")
        for expected in ("positive", "negative")
    }
    if any(value != 10 for value in counts.values()):
        raise ValueError(f"Panel stratification mismatch: {counts}")
    all_ids = [row["case_id"] for row in cases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Panel case identifiers are not unique")

    result_rows = _collect_predictions(fields, cases, scratch_dir)
    blind_rows = []
    answer_rows = []
    for case, result in zip(cases, result_rows, strict=True):
        expected_target_detected = case["example_id"] in json.loads(
            result["matched_example_ids_json"]
        )
        uncertainty = "Expected class is controlled by construction and has not been independently human-verified; scientific review is pending."
        blind_rows.append(
            {
                "case_id": case["case_id"],
                "benchmark": case["benchmark_name"],
                "benchmark_example_id": case["example_id"],
                "corpus_source_or_synthetic_origin": "Synthetic controlled document; no post-D1 corpus row was used.",
                "original_excerpt": case["original_excerpt"],
                "matching_excerpt": case["matching_excerpt"],
                "matcher_prediction": result["prediction"],
                "matched_example_ids_json": result["matched_example_ids_json"],
                "matched_fields_json": result["matched_fields_json"],
                "decision_rules_json": result["decision_rules_json"],
                "matched_spans_json": result["matched_spans_json"],
                "anchor_evidence_json": result["anchor_evidence_json"],
                "uncertainty": uncertainty,
                "scientist_review_label": "",
            }
        )
        answer_rows.append(
            {
                "case_id": case["case_id"],
                "benchmark": case["benchmark_name"],
                "benchmark_example_id": case["example_id"],
                "expected_class": case["expected_class"],
                "expected_relationship": case["expected_relationship"],
                "control_kind": case["control_kind"],
                "known_construction_provenance": case["known_construction_provenance"],
                "synthetic_document_sha256": case["document_sha256"],
                "matcher_prediction": result["prediction"],
                "expected_target_example_detected": expected_target_detected,
                "matched_example_ids_json": result["matched_example_ids_json"],
            }
        )

    blind_columns = list(blind_rows[0])
    answer_columns = list(answer_rows[0])
    _write_csv(output_dir / "review_panel_blind.csv", blind_rows, blind_columns)
    _write_csv(output_dir / "review_panel_answer_key.csv", answer_rows, answer_columns)
    table = pa.Table.from_pylist(result_rows, schema=RESULT_SCHEMA)
    pq.write_table(table, output_dir / "fixture_results.parquet", compression="zstd")

    positive_detected = sum(
        row["expected_class"] == "positive" and row["expected_target_example_detected"]
        for row in answer_rows
    )
    negative_clean = sum(
        row["expected_class"] == "negative"
        and row["matcher_prediction"] == "no_candidate"
        for row in answer_rows
    )
    mismatch_ids = [
        row["case_id"]
        for row in answer_rows
        if (
            row["expected_class"] == "positive"
            and not row["expected_target_example_detected"]
        )
        or (
            row["expected_class"] == "negative"
            and row["matcher_prediction"] != "no_candidate"
        )
    ]
    notes = f"""# C1-BD2.5 independent calibration panel (provisional)

## Panel design

- 40 deterministic cases, with 10 expected positives and 10 expected hard negatives per benchmark.
- All benchmark example IDs are disjoint from IDs used by the existing BD2 fixtures. Belebele panel cases also use distinct passage IDs.
- All cases are controlled synthetic documents built from pinned benchmark excerpts or written synthetic controls. No real post-D1 corpus positives or negatives were selected, and no source-family claims can be made from this panel.
- The blinded file omits the expected class and construction rationale. The separate answer key records the class, relationship, provenance, and synthetic document checksum. Every scientist-review field is empty.
- Expected classes are known from deterministic construction but have not been independently human-verified. The panel is a review packet, not scientific sign-off or a population precision/recall estimate.

Positive controls cover exact complete items and contexts/passages, question plus answer, embedded material, case/punctuation changes, small insertions and deletions, partial contiguous overlap, and short distinctive questions/items. Negative controls cover answer-only text, generic Portuguese, common multiple-choice instructions, theme-adjacent different facts, bibliography-like text, shared proverbs, distinct entities sharing a template, repeated boilerplate, and benchmark fragments separated through long synthetic documents.

## Matcher predictions against construction labels

- CALAME-PT: 10 positive controls; {sum(row["expected_class"] == "positive" and row["benchmark"] == "calame_pt" for row in answer_rows)} negative controls.
- Belebele por_Latn: 10 positive controls; {sum(row["expected_class"] == "negative" and row["benchmark"] == "belebele_por_latn" for row in answer_rows)} negative controls.
- Positive target example detected: {positive_detected}/20.
- Negative controls with no matcher candidate: {negative_clean}/20.
- Construction-label mismatches requiring review: {len(mismatch_ids)} ({", ".join(mismatch_ids) if mismatch_ids else "none"}).
- CALAME snapshot inventory remains 2,076 rows; row 718 has a whitespace-only target and is retained in the inventory but excluded from a future accuracy denominator, giving a 2,075-row evaluable denominator convention.

The result file contains matcher predictions and offsets only; the answer key is separate from the primary review view. CALAME generated-source and Belebele upstream passage-rights/provenance caveats remain unresolved limitations.

## Review decision

Scientist labels and policy conclusions are pending. The candidate policy remains `freeze_for_bd3=false`. Do not treat this constructed panel as evidence that the corpus-wide candidate threshold is calibrated.
"""
    (output_dir / "review_panel_provisional_notes.md").write_text(
        notes, encoding="utf-8"
    )

    manifest = _make_manifest(output_dir, snapshot_check, cases, result_rows)
    manifest_path = output_dir / "preflight_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    verify_artifact_checksums(output_dir)
    return manifest


def verify_artifact_checksums(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "preflight_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Preflight artifact size mismatch: {name}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Preflight artifact checksum mismatch: {name}")
    return {
        "status": "PREFLIGHT_ARTIFACT_CHECKSUMS_VERIFIED",
        "artifacts": len(manifest["artifacts"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            result = verify_artifact_checksums(args.output_dir)
        else:
            result = build_panel(args.output_dir, args.snapshot_dir, args.scratch_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
