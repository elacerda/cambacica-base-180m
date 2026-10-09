"""Deterministic synthetic and pinned-example calibration controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .matcher import CorpusDocument, MatchField, tokenize_with_offsets


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    split: str
    expected_match: bool
    document: CorpusDocument
    rationale: str
    expected_example_id: str | None = None


DEV_PASSAGE = (
    "Em 1987, a engenheira Lídia Seranduva instalou um observatório de marés na ilha de "
    "Pedra Clara. O equipamento registrava a altura da água a cada quinze minutos e "
    "enviava os números por rádio para uma estação costeira. Durante o primeiro inverno, "
    "uma tempestade deslocou o marco de referência, mas a equipe recuperou as medições "
    "comparando cadernos, fotografias e horários anotados pelos pescadores. A série "
    "revelou que a enseada recebia correntes diferentes antes do amanhecer e depois da "
    "passagem de navios cargueiros. Em vez de publicar uma conclusão apressada, os "
    "pesquisadores mantiveram os registros originais, explicaram cada correção e "
    "convidaram outras equipes a repetir a observação durante a estação seguinte."
)
DEV_QUESTION = "Qual instrumento registrava a altura da água na ilha de Pedra Clara?"
DEV_ANSWER = "Um observatório de marés instalado pela engenheira Lídia Seranduva."
DEV_SHORT_QUESTION = "Qual era o nome do observatório de marés?"

HOLDOUT_PASSAGE = (
    "No vale de Nacarim, a pesquisadora Joana Vilar catalogou líquens sobre rochas "
    "expostas ao vento salino. Ela separou as amostras por altitude, fotografou cada "
    "colônia e guardou fragmentos em envelopes numerados. Após seis meses, a equipe "
    "comparou as cores com a umidade medida em pequenas estações meteorológicas. As "
    "diferenças não indicavam uma espécie nova; mostravam que o mesmo organismo crescia "
    "mais lentamente nas encostas secas. O relatório descreveu o método, publicou as "
    "tabelas completas e registrou as limitações do equipamento usado naquele inverno."
)
HOLDOUT_QUESTION = (
    "O que explicava a diferença de crescimento dos líquens nas encostas?"
)
HOLDOUT_ANSWER = "A menor umidade nas encostas secas."

TAXONOMY_PASSAGE = (
    "A mariposa Argidia clarae pertence a um gênero descrito no planalto de Itacuri. "
    "Os exemplares examinados tinham asas estreitas e manchas claras junto à margem. "
    "As observações foram feitas em períodos distintos, e a descrição identifica as "
    "características que diferenciam a espécie de outros lepidópteros. "
    "Referências: Almeida, Catálogo de insetos do sul, páginas 12 a 19; "
    "Barreto, Estudos de lepidópteros, volume 4; Costa, Fauna regional, páginas 80 a 94."
)
SHARED_REFERENCES = (
    "Referências: Almeida, Catálogo de insetos do sul, páginas 12 a 19; "
    "Barreto, Estudos de lepidópteros, volume 4; Costa, Fauna regional, páginas 80 a 94."
)


def _field(
    example_id: str,
    role: str,
    text: str,
    *,
    matchable: bool = True,
    benchmark_name: str = "synthetic_fixture",
) -> MatchField:
    return MatchField(
        benchmark_name=benchmark_name,
        example_id=example_id,
        source_row_id=f"fixture:{example_id}",
        source_file_sha256=hashlib.sha256(b"c1-bd2-synthetic-fixtures-v1").hexdigest(),
        source_category=None,
        field_id=f"{example_id}:{role}",
        field_role=role,
        original_text=text,
        matchable=matchable,
    )


def synthetic_fields() -> list[MatchField]:
    """Return distinct dev and held-out items plus the D2d bibliography case."""
    return [
        _field("synthetic-dev-marés", "passage", DEV_PASSAGE),
        _field("synthetic-dev-marés", "question", DEV_QUESTION),
        _field(
            "synthetic-dev-marés",
            "question_plus_answer",
            f"{DEV_QUESTION} {DEV_ANSWER}",
        ),
        _field(
            "synthetic-dev-marés",
            "complete_item",
            f"{DEV_PASSAGE}\n{DEV_QUESTION}\nA. {DEV_ANSWER}\nB. A escala lunar semanal.",
        ),
        _field("synthetic-dev-marés", "correct_answer", DEV_ANSWER, matchable=False),
        _field("synthetic-dev-short", "question", DEV_SHORT_QUESTION),
        _field(
            "synthetic-dev-short",
            "question_plus_answer",
            f"{DEV_SHORT_QUESTION} {DEV_ANSWER}",
        ),
        _field("synthetic-dev-short", "correct_answer", DEV_ANSWER, matchable=False),
        _field("synthetic-dev-taxonomy", "passage", TAXONOMY_PASSAGE),
        _field("synthetic-holdout-líquen", "passage", HOLDOUT_PASSAGE),
        _field("synthetic-holdout-líquen", "question", HOLDOUT_QUESTION),
        _field(
            "synthetic-holdout-líquen",
            "question_plus_answer",
            f"{HOLDOUT_QUESTION} {HOLDOUT_ANSWER}",
        ),
        _field(
            "synthetic-holdout-líquen",
            "correct_answer",
            HOLDOUT_ANSWER,
            matchable=False,
        ),
    ]


def _case(
    case_id: str,
    split: str,
    text: str,
    expected: bool,
    rationale: str,
    expected_example_id: str | None = None,
    source: str = "synthetic_fixture",
) -> FixtureCase:
    return FixtureCase(
        case_id=case_id,
        split=split,
        expected_match=expected,
        rationale=rationale,
        expected_example_id=expected_example_id,
        document=CorpusDocument(
            doc_id=f"fixture:{case_id}",
            text=text,
            source_shard=f"fixtures/{split}",
            source=source,
            source_row_ordinal=None,
            input_manifest_sha256=None,
        ),
    )


def _token_shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = tuple(token.text for token in tokenize_with_offsets(text))
    return {tokens[index : index + size] for index in range(len(tokens) - size + 1)}


def _near_identical_context(left: str, right: str) -> bool:
    left_tokens = tuple(token.text for token in tokenize_with_offsets(left))
    right_tokens = tuple(token.text for token in tokenize_with_offsets(right))
    if left_tokens == right_tokens:
        return True
    left_long = _token_shingles(left, 13)
    right_long = _token_shingles(right, 13)
    if left_long & right_long:
        return True
    left_short = _token_shingles(left, 5)
    right_short = _token_shingles(right, 5)
    union = left_short | right_short
    return bool(union) and len(left_short & right_short) / len(union) > 0.5


def synthetic_cases() -> list[FixtureCase]:
    """Positive and negative controls; held-out text uses separate item families."""
    edited = DEV_PASSAGE.replace("a cada quinze minutos", "a cada dezesseis minutos")
    edited = edited.replace("A série revelou", "A série local revelou")
    shortened = DEV_PASSAGE.replace("Durante o primeiro inverno, ", "")
    return [
        _case(
            "dev_exact_question",
            "development",
            f"Arquivo de campo. {DEV_QUESTION} A anotação veio do caderno azul.",
            True,
            "Exact benchmark question embedded in unrelated document text.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_exact_complete_item",
            "development",
            f"{DEV_PASSAGE}\n{DEV_QUESTION}\nA. {DEV_ANSWER}\nB. A escala lunar semanal.",
            True,
            "The full benchmark passage, question, and answer choices appear verbatim.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_embedded_full_passage",
            "development",
            f"Boletim costeiro de 1992. {DEV_PASSAGE} Registro transcrito sem alterações.",
            True,
            "Full benchmark passage embedded in a longer document.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_question_plus_answer",
            "development",
            f"Pergunta arquivada: {DEV_QUESTION} Resposta: {DEV_ANSWER}",
            True,
            "The question and its correct answer appear together.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_case_and_punctuation",
            "development",
            f"{DEV_QUESTION.upper()}!!!",
            True,
            "Case and punctuation changes preserve normalized tokens.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_small_insertions_and_deletions",
            "development",
            f"Cópia revista: {edited} {shortened}",
            True,
            "Two long partial copies contain small insertions and deletions.",
            "synthetic-dev-marés",
        ),
        _case(
            "dev_short_distinctive_question",
            "development",
            DEV_SHORT_QUESTION,
            True,
            "Exact short distinctive question is handled separately from the 50-token rule.",
            "synthetic-dev-short",
        ),
        _case(
            "dev_answer_word_only",
            "development",
            "Resposta isolada: Nacarim.",
            False,
            "An answer-like standalone word must not create a hit.",
        ),
        _case(
            "dev_generic_multiple_choice_instructions",
            "development",
            "Assinale a alternativa correta. Escolha somente uma resposta e marque a letra correspondente.",
            False,
            "Generic multiple-choice directions are not benchmark evidence.",
        ),
        _case(
            "dev_common_portuguese_expression",
            "development",
            "De vez em quando, as pessoas observam o que acontece ao redor e conversam sobre isso.",
            False,
            "Common Portuguese expressions do not form distinctive overlap.",
        ),
        _case(
            "dev_same_topic_different_content",
            "development",
            "Uma equipe costeira estudou peixes em um estuário tropical e publicou medições de salinidade feitas durante o verão.",
            False,
            "Topical similarity without copied benchmark wording is not a hit.",
        ),
        _case(
            "dev_taxonomy_shared_references_a",
            "development",
            "A mariposa Argidia clarae tem asas estreitas, manchas claras e habitat de altitude. "
            + SHARED_REFERENCES,
            False,
            "Distinct taxonomy entities may share nearly identical references (D2d pattern).",
            "synthetic-dev-taxonomy",
        ),
        _case(
            "dev_taxonomy_shared_references_b",
            "development",
            "A mariposa Gnamptonychia escura possui antenas largas e foi observada em bosque úmido. "
            + SHARED_REFERENCES,
            False,
            "A distinct entity shares the bibliography but not the entity-specific description.",
            "synthetic-dev-taxonomy",
        ),
        _case(
            "dev_same_structure_different_entity",
            "development",
            "O gênero Mavisolae foi descrito no vale de Irapuã. Os exemplares apresentam asas longas e linhas escuras; "
            "a observação ocorreu em florestas de altitude e não identificou relação com Argidia.",
            False,
            "A repeated structural template with a different entity is not sufficient.",
        ),
        _case(
            "dev_similar_question_changed_facts",
            "development",
            "Qual instrumento media a temperatura do solo em Pedra Clara? Um termômetro registrava a variação diária.",
            False,
            "A related question has different facts and answer wording.",
        ),
        _case(
            "holdout_embedded_passage",
            "heldout",
            f"Relatório de campo em Nacarim: {HOLDOUT_PASSAGE} Fim do trecho.",
            True,
            "Held-out exact passage from a separate synthetic item family.",
            "synthetic-holdout-líquen",
        ),
        _case(
            "holdout_question_answer",
            "heldout",
            f"{HOLDOUT_QUESTION} {HOLDOUT_ANSWER}",
            True,
            "Held-out question and correct answer appear together.",
            "synthetic-holdout-líquen",
        ),
        _case(
            "holdout_same_topic_paraphrase",
            "heldout",
            "No vale de Nacarim, amostras de organismos cresceram mais devagar em encostas com pouca água. "
            "A equipe comparou fotografias e condições meteorológicas antes de publicar a análise.",
            False,
            "Same broad topic with independently worded content.",
        ),
        _case(
            "holdout_short_common_phrase",
            "heldout",
            "A equipe analisou os dados e apresentou os resultados no relatório final.",
            False,
            "A short generic phrase has no benchmark-specific anchor.",
        ),
        _case(
            "holdout_different_factual_answer",
            "heldout",
            "O que explicava as mudanças dos líquens? A quantidade de luz solar em cada encosta.",
            False,
            "A similar question form with an altered factual answer should not match.",
        ),
    ]


def benchmark_derived_cases(
    registry_rows: list[dict[str, Any]],
) -> list[FixtureCase]:
    """Pick disjoint real benchmark items using stable hash ordering."""

    def rank(row: dict[str, Any]) -> tuple[str, str]:
        return hashlib.sha256(row["example_id"].encode("ascii")).hexdigest(), row[
            "example_id"
        ]

    calame_hand = sorted(
        (
            row
            for row in registry_rows
            if row["benchmark_name"] == "calame_pt"
            and row["source_category"] == "handwritten"
        ),
        key=rank,
    )
    calame_generated = sorted(
        (
            row
            for row in registry_rows
            if row["benchmark_name"] == "calame_pt"
            and row["source_category"] == "generated"
        ),
        key=rank,
    )
    belebele_by_passage: dict[str, list[dict[str, Any]]] = {}
    for row in registry_rows:
        if row["benchmark_name"] == "belebele_por_latn":
            raw = json.loads(row["raw_row_json"])
            belebele_by_passage.setdefault(str(raw["link"]), []).append(row)
    belebele_rows = sorted(
        (min(items, key=rank) for items in belebele_by_passage.values()), key=rank
    )
    if not calame_hand or not calame_generated or len(belebele_rows) < 2:
        raise ValueError(
            "Snapshot is too small for disjoint benchmark-derived fixtures"
        )
    dev_calame = calame_hand[0]
    holdout_calame = next(
        (
            row
            for row in calame_generated
            if not _near_identical_context(dev_calame["context"], row["context"])
        ),
        None,
    )
    if holdout_calame is None:
        raise ValueError(
            "Cannot choose disjoint CALAME calibration families without a near-copy"
        )
    dev_belebele = belebele_rows[0]
    dev_passage = dev_belebele["passage"]
    holdout_belebele = next(
        (
            row
            for row in belebele_rows[1:]
            if row["source_row_id"].rsplit("#q", 1)[0]
            != dev_belebele["source_row_id"].rsplit("#q", 1)[0]
            and not _near_identical_context(dev_passage, row["passage"])
        ),
        None,
    )
    if holdout_belebele is None:
        raise ValueError("Cannot choose disjoint Belebele passages without a near-copy")
    selected = [dev_calame, dev_belebele, holdout_calame, holdout_belebele]
    if len({row["example_id"] for row in selected}) != len(selected):
        raise ValueError("Calibration and held-out benchmark examples overlap")
    if _near_identical_context(selected[0]["context"], selected[2]["context"]):
        raise ValueError("CALAME development/held-out controls share a near-copy")
    if _near_identical_context(selected[1]["passage"], selected[3]["passage"]):
        raise ValueError("Belebele development/held-out controls share a near-copy")

    cases: list[FixtureCase] = []
    cases.append(
        _case(
            "real_dev_calame_handwritten_complete_item",
            "development",
            f"Registro de avaliação. {calame_hand[0]['context']} {calame_hand[0]['target_word']} Fim.",
            True,
            "Pinned CALAME handwritten row, complete context and target.",
            calame_hand[0]["example_id"],
            source="benchmark_derived_control",
        )
    )
    dev_bele = dev_belebele
    cases.append(
        _case(
            "real_dev_belebele_embedded_passage",
            "development",
            f"Arquivo público. {dev_bele['passage']} Nota editorial.",
            True,
            "Pinned Belebele Portuguese passage embedded in a wrapper.",
            dev_bele["example_id"],
            source="benchmark_derived_control",
        )
    )
    cases.append(
        _case(
            "real_dev_belebele_question_answer",
            "development",
            f"{dev_bele['question']} {dev_bele['answer']}",
            True,
            "Pinned Belebele question plus its one-indexed correct option text.",
            dev_bele["example_id"],
            source="benchmark_derived_control",
        )
    )
    cases.append(
        _case(
            "real_dev_answer_only",
            "development",
            dev_bele["answer"],
            False,
            "The correct answer text alone is explicitly non-triggering.",
            source="benchmark_derived_control",
        )
    )
    cases.append(
        _case(
            "real_holdout_calame_generated_complete_item",
            "heldout",
            f"Cópia de avaliação: {holdout_calame['context']} {holdout_calame['target_word']}",
            True,
            "Pinned CALAME generated row, separate from handwritten development item.",
            holdout_calame["example_id"],
            source="benchmark_derived_control",
        )
    )
    hold_bele = holdout_belebele
    cases.append(
        _case(
            "real_holdout_belebele_embedded_passage",
            "heldout",
            f"Trecho reproduzido: {hold_bele['passage']} Referência encerrada.",
            True,
            "Pinned Belebele passage from a different source link than development.",
            hold_bele["example_id"],
            source="benchmark_derived_control",
        )
    )
    return cases
