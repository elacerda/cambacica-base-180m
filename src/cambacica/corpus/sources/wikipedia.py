"""Portuguese Wikipedia sampler for Gate C1.

Streams articles from wikimedia/wikipedia (subset 20231101.pt) using Hugging Face
streaming, applying deterministic reservoir sampling on MediaWiki page IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import datasets

from cambacica.corpus.manifest import ProvenanceManifest
from cambacica.corpus.sampling import DeterministicReservoirSampler
from cambacica.corpus.schema import validate_and_normalize
from cambacica.corpus.sources.base import (
    BaseSourceSampler,
    ensure_user_hf_cache,
    resolve_hf_commit_sha,
)


logger = logging.getLogger(__name__)


class WikipediaSampler(BaseSourceSampler):
    """Sampler for Portuguese Wikipedia.

    Supports:
    - 'representative': Collects encyclopedic articles from a bounded stream prefix
      using deterministic reservoir sampling over MediaWiki page IDs.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(source_name="wikipedia_pt", config=config)
        self.canonical_id: str = "wikimedia/wikipedia"
        self.revision: str = "20231101.pt"

    def plan(
        self,
        mode: str = "representative",
        size: int = 10000,
        max_stream_items: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 10000
            Target number of articles to sample.
        max_stream_items : int or None, optional
            Maximum stream items to evaluate.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        dict
            Dry-run execution plan.
        """
        if max_stream_items is None:
            max_stream_items = max(size * 3, 30000)
        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        est_mb = (max_stream_items * 5) / 1024
        return {
            "source": self.source_name,
            "mode": mode,
            "target_size": size,
            "upstream_identifier": self.canonical_id,
            "upstream_revision": self.revision,
            "upstream_commit_sha": commit_sha,
            "upstream_configuration": self.revision,
            "population_scope": (
                "~1,183,410 articles (wikimedia/wikipedia 20231101.pt split 'train')"
            ),
            "sampling_frame": (
                f"bounded_stream_prefix (first {max_stream_items:,} articles from Hugging Face streaming)"
            ),
            "selected_partitions": [
                f"{self.canonical_id} [{self.revision}] train split"
            ],
            "safety_limits": {
                "max_stream_items": max_stream_items,
            },
            "estimated_transfer": f"~{est_mb:.1f} MB (bounded stream scan)",
        }

    def sample(
        self,
        mode: str = "representative",
        size: int = 10000,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        max_stream_items: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic sampling from Portuguese Wikipedia.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 10000
            Target number of articles to sample.
        seed : int, default 42
            Deterministic random seed.
        output_dir : Path or str or None, optional
            Output destination directory.
        max_stream_items : int or None, optional
            Maximum stream rows to inspect before finalizing reservoir.
            Defaults to size * 3 for safety and efficiency.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Path to Parquet file and metadata manifest.
        """
        ensure_user_hf_cache()

        if output_dir is None:
            output_dir = Path("data/samples/gate_c1/wikipedia_pt")
        else:
            output_dir = Path(output_dir)

        if max_stream_items is None:
            max_stream_items = max(size * 3, 30000)

        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)

        ds = datasets.load_dataset(
            self.canonical_id,
            self.revision,
            split="train",
            streaming=True,
        )

        count = 0
        bytes_read = 0
        stopping_reason = "stream_exhausted"

        for row in ds:
            count += 1
            text = row.get("text", "")
            title = row.get("title", "")
            if title and not text.startswith(title):
                full_text = f"{title}\n\n{text}"
            else:
                full_text = text

            bytes_read += len(full_text.encode("utf-8")) + 200

            if not full_text or len(full_text.strip()) < 50:
                continue

            page_id = str(row.get("id", count))
            url = row.get("url")

            raw_doc = {
                "text": full_text,
                "source": "wikipedia_pt",
                "source_revision": self.revision,
                "subset": "articles",
                "original_id": page_id,
                "original_url": url,
                "license": "CC BY-SA 3.0/4.0 & GFDL",
                "language": "pt",
                "language_score": 1.0,
                "variety": None,  # Pluricentric mix; do not fabricate PT-BR/PT-PT
                "quality_score": None,
                "publication_date": None,
                "domain_category": "encyclopedic",
            }

            norm_doc = validate_and_normalize(raw_doc)
            reservoir.add(page_id, norm_doc)

            if len(reservoir) >= size and count >= max_stream_items:
                stopping_reason = "target_size_reached_and_stream_limit_met"
                break

        documents = reservoir.get_sample()
        if len(documents) >= size and stopping_reason == "stream_exhausted":
            stopping_reason = "target_size_reached"

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
            upstream_configuration=self.revision,
            population_scope=(
                "~1,183,410 articles (wikimedia/wikipedia 20231101.pt split 'train')"
            ),
            sampling_frame=(
                f"bounded_stream_prefix (first {max_stream_items:,} articles from Hugging Face streaming)"
            ),
            records_examined=count,
            bytes_read=bytes_read,
            stopping_reason=stopping_reason,
        )
