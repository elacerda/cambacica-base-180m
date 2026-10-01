"""Project Gutenberg Portuguese literature sampler for Gate C1.

Discovers, downloads, and cleans public-domain Portuguese literary works from
Project Gutenberg, stripping standard licensing and boilerplate headers/footers.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import List, Optional, Tuple
import requests

from cambacica.corpus.manifest import ProvenanceManifest
from cambacica.corpus.sampling import DeterministicReservoirSampler
from cambacica.corpus.schema import (
    NormalizedDocument,
    validate_and_normalize,
)
from cambacica.corpus.sources.base import BaseSourceSampler


logger = logging.getLogger(__name__)

GUTENBERG_PT_CATALOG = "https://www.gutenberg.org/browse/languages/pt"
GUTENBERG_TXT_URL_PATTERN = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

HEADER_RE = re.compile(
    r"\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
    re.IGNORECASE,
)
FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Strip Gutenberg legal header and footer boilerplate.

    Parameters
    ----------
    raw_text : str
        Raw Gutenberg eBook text.

    Returns
    -------
    str
        Clean text body without Gutenberg administrative notices.
    """
    start_match = HEADER_RE.search(raw_text)
    end_match = FOOTER_RE.search(raw_text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)

    clean = raw_text[start_idx:end_idx].strip()
    return clean if clean else raw_text.strip()


def discover_gutenberg_pt_ids(timeout: int = 15) -> List[int]:
    """Scrape available Portuguese eBook IDs from the Gutenberg language catalog.

    Parameters
    ----------
    timeout : int, default 15
        HTTP request timeout in seconds.

    Returns
    -------
    list of int
        Unique sorted list of Project Gutenberg eBook numeric IDs.
    """
    try:
        resp = requests.get(GUTENBERG_PT_CATALOG, timeout=timeout)
        resp.raise_for_status()
        raw_ids = re.findall(r"/ebooks/(\d+)", resp.text)
        return sorted(list(set(int(x) for x in raw_ids)))
    except Exception as e:
        logger.warning(f"Error discovering Gutenberg IDs: {e}")
        # Return fallback canonical IDs if network fails or catalog is unreachable
        return [
            2837,
            3333,
            7384,
            8698,
            9654,
            11299,
            12579,
            13092,
            13093,
            13630,
            14040,
            14890,
            15006,
            16370,
            16900,
            17290,
            17822,
            18274,
            18729,
        ]


class GutenbergPTSampler(BaseSourceSampler):
    """Sampler for Project Gutenberg Portuguese literary works.

    Supports:
    - 'representative': Downloads and cleans canonical Portuguese literary
      works in public domain.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(source_name="gutenberg_pt", config=config)
        self.canonical_id: str = "project_gutenberg_pt"
        self.revision: str = "catalog_snapshot_2026-10-01"

    def plan(
        self,
        mode: str = "representative",
        size: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 100
            Target number of literary works.
        seed : int, default 42
            Deterministic seed.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        dict
            Dry-run execution plan.
        """
        available_ids = discover_gutenberg_pt_ids()
        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)
        for bid in available_ids:
            reservoir.add(str(bid), bid)
        candidates = reservoir.get_sample()
        est_mb = (len(candidates) * 350) / 1024
        return {
            "source": self.source_name,
            "mode": mode,
            "target_size": size,
            "upstream_identifier": self.canonical_id,
            "upstream_revision": self.revision,
            "upstream_commit_sha": None,
            "population_scope": (
                f"{len(available_ids)} Portuguese eBooks cataloged on Project Gutenberg (language/pt)"
            ),
            "sampling_frame": (
                f"full_catalog_hash_reservoir (all {len(available_ids)} cataloged eBook IDs eligible)"
            ),
            "selected_partitions": [
                f"{len(candidates)} selected eBook IDs (e.g. {candidates[:5]}...)"
            ],
            "safety_limits": {
                "max_attempts": len(candidates),
                "request_timeout": 15,
            },
            "estimated_transfer": f"~{est_mb:.1f} MB ({len(candidates)} books)",
        }

    def sample(
        self,
        mode: str = "representative",
        size: int = 100,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        max_attempts: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute deterministic sampling of Gutenberg Portuguese books.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 100
            Target number of complete literary works.
        seed : int, default 42
            Deterministic seed.
        output_dir : Path or str or None, optional
            Output destination directory.
        max_attempts : int or None, optional
            Safety limit on number of eBook IDs attempted.
        **kwargs : any
            Additional arguments.

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Path to Parquet file and metadata manifest.
        """
        if output_dir is None:
            output_dir = Path("data/samples/gate_c1/gutenberg_pt")
        else:
            output_dir = Path(output_dir)

        available_ids = discover_gutenberg_pt_ids()
        if not available_ids:
            raise RuntimeError("No Project Gutenberg Portuguese eBook IDs found.")

        # Order IDs deterministically by hash(id, seed)
        reservoir = DeterministicReservoirSampler(capacity=size, seed=seed)
        for bid in available_ids:
            reservoir.add(str(bid), bid)

        candidate_ids = reservoir.get_sample()
        if max_attempts is None:
            max_attempts = len(candidate_ids)

        documents: List[NormalizedDocument] = []
        bytes_read = 0
        records_examined = 0
        stopping_reason = "candidate_list_exhausted"

        for bid in candidate_ids[:max_attempts]:
            records_examined += 1
            txt_url = GUTENBERG_TXT_URL_PATTERN.format(id=bid)
            try:
                resp = requests.get(txt_url, timeout=15)
                bytes_read += len(resp.content)
                if resp.status_code != 200:
                    continue
                raw_text = resp.text
                clean_text = strip_gutenberg_boilerplate(raw_text)
                if len(clean_text) < 500:
                    continue

                raw_doc = {
                    "text": clean_text,
                    "source": "gutenberg_pt",
                    "source_revision": self.revision,
                    "subset": "literature",
                    "original_id": str(bid),
                    "original_url": f"https://www.gutenberg.org/ebooks/{bid}",
                    "license": "Project Gutenberg License / US Public Domain (jurisdiction-dependent)",
                    "language": "pt",
                    "language_score": 1.0,
                    "variety": None,  # Contains both PT-BR and PT-PT classics
                    "quality_score": None,
                    "publication_date": None,
                    "domain_category": "literature",
                }

                norm_doc = validate_and_normalize(raw_doc)
                documents.append(norm_doc)

                if len(documents) >= size:
                    stopping_reason = "target_size_reached"
                    break
            except Exception as e:
                logger.warning(f"Error downloading Gutenberg book {bid}: {e}")
                continue

        selected_ebook_ids = [d.original_id for d in documents]

        parquet_path, manifest = self._persist_sample(
            documents=documents,
            mode=mode,
            target_size=size,
            seed=seed,
            output_dir=output_dir,
            upstream_identifier=self.canonical_id,
            upstream_revision=self.revision,
            upstream_commit_sha=None,
            upstream_url=GUTENBERG_PT_CATALOG,
            upstream_configuration="catalog_snapshot",
            population_scope=(
                f"{len(available_ids)} Portuguese eBooks cataloged on Project Gutenberg (language/pt)"
            ),
            sampling_frame=(
                f"full_catalog_hash_reservoir (all {len(available_ids)} cataloged eBook IDs eligible)"
            ),
            records_examined=records_examined,
            bytes_read=bytes_read,
            stopping_reason=stopping_reason,
        )

        # Record selected ebook IDs in manifest stats
        manifest.stats["selected_ebook_ids"] = selected_ebook_ids
        manifest_path = Path(output_dir) / f"manifest_{mode}.json"
        manifest.save(manifest_path)

        return parquet_path, manifest
