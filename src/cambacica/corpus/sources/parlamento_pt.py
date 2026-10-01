"""ParlamentoPT sampler for Gate C1.

Streams European Portuguese parliamentary debate records from PORTULAN/parlamento-pt
without downloading the full 2.7 GB text archive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests

from cambacica.corpus.manifest import ProvenanceManifest
from cambacica.corpus.sampling import DeterministicReservoirSampler
from cambacica.corpus.schema import (
    NormalizedDocument,
    validate_and_normalize,
)
from cambacica.corpus.sources.base import (
    BaseSourceSampler,
    resolve_hf_commit_sha,
)


logger = logging.getLogger(__name__)

PARLAMENTO_URL = (
    "https://huggingface.co/datasets/PORTULAN/parlamento-pt/resolve/main/train.txt"
)


class ParlamentoPTSampler(BaseSourceSampler):
    """Sampler for European Portuguese parliamentary debates (PORTULAN/parlamento-pt).

    Supports:
    - 'representative': Collects debate documents in native European Portuguese
      (PT-PT) using deterministic reservoir sampling over a bounded stream prefix.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(source_name="parlamento_pt", config=config)
        self.canonical_id: str = "PORTULAN/parlamento-pt"
        self.revision: str = "main"

    def plan(
        self,
        mode: str = "representative",
        size: int = 10000,
        min_length: int = 0,
        max_stream_lines: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 10000
            Target number of debate documents.
        min_length : int, default 0
            Minimum character filter length (disabled by default).
        max_stream_lines : int or None, optional
            Maximum lines to read from the stream prefix.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        dict
            Dry-run execution plan.
        """
        if max_stream_lines is None:
            max_stream_lines = max(size * 4, 40000)
        commit_sha = resolve_hf_commit_sha(
            self.canonical_id, revision=self.revision
        )
        est_bytes = max_stream_lines * 250
        return {
            "source": self.source_name,
            "mode": mode,
            "target_size": size,
            "upstream_identifier": self.canonical_id,
            "upstream_revision": self.revision,
            "upstream_commit_sha": commit_sha,
            "population_scope": (
                "~11.5M debate interventions in PORTULAN/parlamento-pt "
                "(train.txt, ~2.7 GB uncompressed)"
            ),
            "sampling_frame": (
                f"bounded_stream_prefix (first {max_stream_lines:,} lines of train.txt)"
            ),
            "selected_partitions": [PARLAMENTO_URL],
            "safety_limits": {
                "max_stream_lines": max_stream_lines,
                "min_length_filter": min_length,
                "timeout_seconds": 30,
            },
            "estimated_transfer": f"~{est_bytes / (1024 * 1024):.1f} MB (bounded stream scan)",
        }

    def sample(
        self,
        mode: str = "representative",
        size: int = 10000,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        max_stream_lines: Optional[int] = None,
        min_length: int = 0,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic sampling from ParlamentoPT.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 10000
            Target number of debate documents.
        seed : int, default 42
            Deterministic seed.
        output_dir : Path or str or None, optional
            Output destination directory.
        max_stream_lines : int or None, optional
            Safety limit on stream lines read.
        min_length : int, default 0
            Optional minimum character length filter (disabled by default).
            When 0, all non-empty valid parliamentary utterances are kept.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Path to Parquet output and metadata manifest.
        """
        if output_dir is None:
            output_dir = Path("data/samples/gate_c1/parlamento_pt")
        else:
            output_dir = Path(output_dir)

        if max_stream_lines is None:
            max_stream_lines = max(size * 4, 40000)

        commit_sha = resolve_hf_commit_sha(
            self.canonical_id, revision=self.revision
        )
        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)

        line_count = 0
        accepted_count = 0
        bytes_read = 0
        stopping_reason = "stream_exhausted"

        # Stream HTTP response line by line safely
        with requests.get(PARLAMENTO_URL, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                line_count += 1
                if raw_line is not None:
                    bytes_read += len(raw_line.encode("utf-8")) + 1

                if not raw_line or not raw_line.strip():
                    continue

                text = raw_line.strip()
                # Optional minimum length quality filter (disabled by default)
                if min_length > 0 and len(text) < min_length:
                    continue

                doc_id = f"parl_pt_{line_count}"
                raw_doc = {
                    "text": text,
                    "source": "parlamento_pt",
                    "source_revision": self.revision,
                    "subset": "debates",
                    "original_id": doc_id,
                    "original_url": (
                        "https://www.parlamento.pt/Cidadania/Paginas/DadosAbertos.aspx"
                    ),
                    "license": "Open Government Data (Portuguese Parliament)",
                    "language": "pt-PT",
                    "language_score": 1.0,
                    "variety": "pt-PT",
                    "quality_score": None,
                    "publication_date": None,
                    "domain_category": "parliamentary_debates",
                }

                norm_doc = validate_and_normalize(raw_doc)
                if reservoir.add(doc_id, norm_doc):
                    accepted_count += 1

                if len(reservoir) >= size and line_count >= max_stream_lines:
                    stopping_reason = "target_size_reached_and_stream_limit_met"
                    break

        documents = reservoir.get_sample()

        return self._persist_sample(
            documents=documents,
            mode=mode,
            target_size=size,
            seed=seed,
            output_dir=output_dir,
            upstream_identifier=self.canonical_id,
            upstream_revision=self.revision,
            upstream_commit_sha=commit_sha,
            upstream_url=f"https://huggingface.co/datasets/{self.canonical_id}",
            upstream_configuration="main",
            population_scope=(
                "~11.5M debate interventions in PORTULAN/parlamento-pt "
                "(train.txt, ~2.7 GB uncompressed)"
            ),
            sampling_frame=(
                f"bounded_stream_prefix (first {max_stream_lines:,} lines of train.txt)"
            ),
            records_examined=line_count,
            bytes_read=bytes_read,
            stopping_reason=stopping_reason,
        )
