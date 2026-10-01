"""Corpus Carolina sampler for Gate C1.

Streams and extracts documents from the Corpus Carolina repository
(carolina-c4ai/corpus-carolina) in TEI P5 XML format, supporting both
representative sampling and diagnostic stratified sampling.

Taxonomy population counts
---------------------------
Source: Corpus Carolina v2.0.1 official release metadata.
Pinned upstream commit: 55e63a519393c70a48dcfa14a558499c6bb0583b

These are the *exact* documented document counts used to derive
proportional allocation weights for representative mode.

dat:  1 074 032  (datasets and other corpora — largest)
wik:    957 501  (wikis)
jud:     38 187  (judicial branch)
uni:     26 409  (university domains)
soc:      8 862  (social media)
leg:      3 982  (legislative branch)
pub:         26  (public domain works)
total: 2 108 999

For N = 10 000 the largest-remainder allocation yields:
dat 5093 | wik 4540 | jud 181 | uni 125 | soc 42 | leg 19 | pub 0
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET

from huggingface_hub import HfFileSystem

from cambacica.corpus.manifest import ProvenanceManifest
from cambacica.corpus.sampling import (
    DeterministicReservoirSampler,
    StratifiedSampler,
    stable_hash64,
)
from cambacica.corpus.schema import (
    NormalizedDocument,
    validate_and_normalize,
)
from cambacica.corpus.sources.base import (
    BaseSourceSampler,
    resolve_hf_commit_sha,
)


logger = logging.getLogger(__name__)

TEI_NS = "{http://www.tei-c.org/ns/1.0}"

TAXONOMY_DIRS: Dict[str, str] = {
    "wik": "corpus/wikis",
    "dat": "corpus/datasets_and_other_corpora",
    "soc": "corpus/social_media",
    "uni": "corpus/university_domains",
    "jud": "corpus/judicial_branch",
    "leg": "corpus/legislative_branch",
    "pub": "corpus/public_domain_works",
}

# Exact document counts from Corpus Carolina v2.0.1 official release metadata.
# Pinned upstream commit: 55e63a519393c70a48dcfa14a558499c6bb0583b
# These values MUST NOT be approximated or invented.
# Update this dict only when the upstream revision changes and new counts are
# published in the corresponding release notes.
CAROLINA_V2_0_1_COMMIT_SHA: str = "55e63a519393c70a48dcfa14a558499c6bb0583b"

TAXONOMY_POPULATION: Dict[str, int] = {
    "dat": 1_074_032,
    "wik":   957_501,
    "jud":    38_187,
    "uni":    26_409,
    "soc":     8_862,
    "leg":     3_982,
    "pub":        26,
}
# Sanity-check: sum must equal the documented total.
assert sum(TAXONOMY_POPULATION.values()) == 2_108_999, (
    "TAXONOMY_POPULATION sum does not match documented total of 2,108,999 for v2.0.1"
)


def _proportional_allocation(
    target: int,
    populations: Dict[str, int],
) -> Dict[str, int]:
    """Allocate a target count proportionally using the largest-remainder method.

    The Hamilton (largest-remainder) method guarantees that:
    - The allocation sums to *exactly* ``target``.
    - No category receives less than 0 documents.
    - The rounding error is distributed to the categories with the largest
      fractional remainders.
    - Rare categories (e.g. ``pub`` with only 26 documents when target=10 000)
      may legitimately receive zero allocation in representative mode.
      Diagnostic mode uses explicit minimum quotas instead.

    Parameters
    ----------
    target : int
        Total number of documents to allocate.
    populations : dict of str to int
        Mapping from category name to upstream population count.

    Returns
    -------
    dict of str to int
        Per-category allocation summing to exactly ``target``.
    """
    total_pop = sum(populations.values())
    if total_pop == 0:
        n = max(1, len(populations))
        base = target // n
        remainder = target - base * n
        result = {k: base for k in populations}
        for i, k in enumerate(populations):
            if i < remainder:
                result[k] += 1
        return result

    # Step 1: compute exact quotas and floor allocations
    quotas = {k: target * v / total_pop for k, v in populations.items()}
    floors = {k: int(q) for k, q in quotas.items()}
    remainders = {k: quotas[k] - floors[k] for k in populations}

    # Step 2: distribute leftover slots to highest-remainder categories
    total_floor = sum(floors.values())
    leftover = target - total_floor
    ranked = sorted(populations.keys(), key=lambda k: remainders[k], reverse=True)
    allocation = dict(floors)
    for k in ranked[:leftover]:
        allocation[k] += 1

    return allocation


class _CountingReader:
    """Reader wrapper that counts bytes read."""

    def __init__(self, raw: Any, counter: List[int]) -> None:
        self.raw = raw
        self.counter = counter

    def read(self, size: int = -1) -> bytes:
        """Read bytes and update counter.

        Parameters
        ----------
        size : int, default -1
            Number of bytes to read.

        Returns
        -------
        bytes
            Read data.
        """
        chunk = self.raw.read(size)
        if chunk:
            self.counter[0] += len(chunk)
        return chunk


def _extract_tei_document(
    elem: ET.Element,
    taxonomy: str,
) -> Optional[dict]:
    """Extract text and metadata fields from a parsed TEI XML element.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        Root TEI element.
    taxonomy : str
        Three-letter taxonomy code (e.g. 'jud', 'leg', 'soc').

    Returns
    -------
    dict or None
        Raw document mapping ready for normalization, or None if empty.
    """
    paragraphs = [
        p.text.strip()
        for p in elem.findall(f".//{TEI_NS}body//{TEI_NS}p")
        if p.text and p.text.strip()
    ]
    if not paragraphs:
        # Fallback to any paragraphs in the document
        paragraphs = [
            p.text.strip()
            for p in elem.findall(f".//{TEI_NS}p")
            if p.text and p.text.strip()
        ]

    full_text = "\n\n".join(paragraphs).strip()
    if not full_text:
        return None

    header = elem.find(f".//{TEI_NS}teiHeader")
    doc_id = None
    title = None
    pub_date = None
    license_tag = None
    url = None

    if header is not None:
        title_el = header.find(f".//{TEI_NS}title")
        if title_el is not None and title_el.text:
            title = title_el.text.strip()
            doc_id = title

        # Check date
        date_el = header.find(f".//{TEI_NS}sourceDesc//{TEI_NS}date")
        if date_el is None:
            date_el = header.find(f".//{TEI_NS}date")
        if date_el is not None and date_el.text:
            pub_date = date_el.text.strip()

        # Check license
        lic_el = header.find(f".//{TEI_NS}license")
        if lic_el is not None and lic_el.text:
            license_tag = lic_el.text.strip()
        else:
            license_tag = "Unspecified / Source-specific (see TEI header)"

        # Check URL in ref
        ref_el = header.find(f".//{TEI_NS}ref")
        if ref_el is not None and "target" in ref_el.attrib:
            url = ref_el.attrib["target"].strip()

    return {
        "text": full_text,
        "source": "carolina",
        "source_revision": "v2.0.1",
        "subset": taxonomy,
        "original_id": doc_id or title,
        "original_url": url,
        "license": license_tag,
        "language": "pt-BR",
        "language_score": 1.0,
        "variety": "pt-BR",
        "quality_score": None,
        "publication_date": pub_date,
        "domain_category": taxonomy,
    }


def iter_carolina_xml_files(
    fs: HfFileSystem,
    taxonomy: str,
) -> List[str]:
    """List and sort available .xml.gz files for a specific Carolina taxonomy.

    Recursively descends into subdirectories (e.g. ``pt-BR/``, ``pt/``) so that
    taxonomies such as *wik* and *dat* — whose shards live one level deeper than
    the taxonomy root — are discovered correctly.

    Parameters
    ----------
    fs : HfFileSystem
        Hugging Face file system instance.
    taxonomy : str
        Taxonomy key ('wik', 'dat', 'jud', 'leg', 'uni', 'soc', 'pub').

    Returns
    -------
    list of str
        Sorted list of HF file paths ending in '.xml.gz'.
    """
    sub_path = TAXONOMY_DIRS.get(taxonomy)
    if not sub_path:
        return []
    full_dir = f"datasets/carolina-c4ai/corpus-carolina/{sub_path}"

    found: List[str] = []

    def _scan(directory: str) -> None:
        try:
            entries = fs.ls(directory, detail=False)
        except Exception as e:
            logger.warning(f"Error listing {directory}: {e}")
            return
        for entry in entries:
            if entry.endswith(".xml.gz"):
                found.append(entry)
            elif not entry.endswith(".sha256") and not entry.endswith(".py") and not entry.endswith(".sh") and not entry.endswith(".rng"):
                # Recurse into sub-directories (e.g. pt-BR/, pt/)
                # Guard against hidden/temp files
                if "." not in entry.split("/")[-1] or entry.split("/")[-1].startswith("."):
                    # Likely a directory – recurse
                    try:
                        if fs.isdir(entry):
                            _scan(entry)
                    except Exception:
                        pass

    _scan(full_dir)
    return sorted(found)



def _select_distributed_shards(
    all_files: List[str],
    n_shards: int,
    seed: int = 42,
) -> List[str]:
    """Select *n_shards* files distributed deterministically over the full list.

    The selection spreads shards evenly across the file list so that early,
    middle, and late shards are all represented.  Within each bucket one file
    is chosen by stable hash so the result is reproducible.

    Parameters
    ----------
    all_files : list of str
        Sorted list of all available shard paths.
    n_shards : int
        Number of shards to select.
    seed : int, default 42
        Deterministic seed.

    Returns
    -------
    list of str
        Selected shard paths, in original sort order.
    """
    n = len(all_files)
    if n == 0:
        return []
    n_shards = min(n_shards, n)
    if n_shards <= 0:
        return []

    step = n / n_shards
    selected = []
    seen: set = set()
    for i in range(n_shards):
        center = int(i * step + step / 2)
        center = min(center, n - 1)
        # Hash-based tie-break within a ±1 window to avoid sequential bias
        candidates = [max(0, center - 1), center, min(n - 1, center + 1)]
        best = min(
            candidates,
            key=lambda idx: stable_hash64(all_files[idx], seed=seed),
        )
        if best not in seen:
            seen.add(best)
            selected.append(all_files[best])

    return sorted(selected, key=lambda p: all_files.index(p))


def stream_carolina_taxonomy(
    fs: HfFileSystem,
    taxonomy: str,
    shard_indices: Optional[List[int]] = None,
    max_files: Optional[int] = None,
    byte_counter: Optional[List[int]] = None,
) -> Iterator[dict]:
    """Stream documents from a Carolina taxonomy over selected shards.

    Parameters
    ----------
    fs : HfFileSystem
        Hugging Face file system instance.
    taxonomy : str
        Taxonomy identifier.
    shard_indices : list of int or None, optional
        Zero-based indices of shards to visit (takes priority over max_files).
    max_files : int or None, optional
        Maximum number of shards to process when shard_indices is None.
    byte_counter : list of int or None, optional
        Single-element list tracking total compressed bytes read.

    Yields
    ------
    dict
        Raw document mapping.
    """
    xml_files = iter_carolina_xml_files(fs, taxonomy)
    if not xml_files:
        return

    if shard_indices is not None:
        selected = [xml_files[i] for i in shard_indices if i < len(xml_files)]
    elif max_files is not None:
        selected = xml_files[:max_files]
    else:
        selected = xml_files

    counter = byte_counter if byte_counter is not None else [0]
    for fpath in selected:
        try:
            with fs.open(fpath, "rb") as gz_raw:
                wrapped_raw = _CountingReader(gz_raw, counter)
                with gzip.GzipFile(fileobj=wrapped_raw) as gz:
                    for _, elem in ET.iterparse(gz, events=("end",)):
                        if elem.tag.endswith("TEI"):
                            doc = _extract_tei_document(elem, taxonomy)
                            elem.clear()
                            if doc:
                                yield doc
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            continue


class CarolinaSampler(BaseSourceSampler):
    """Sampler for Corpus Carolina.

    Supports:
    - 'representative': Samples documents proportionally across taxonomies
      using documented population weights.  Within each taxonomy, shards are
      distributed across the full file list so that both wik and dat (the
      dominant categories) are properly represented.
    - 'diagnostic': Stratified sampling enforcing explicit quotas per taxonomy.
      Continues through additional deterministic shards until each quota is
      satisfied or the taxonomy is truly exhausted.  Underfill is reported
      explicitly and reflected in the manifest stopping reason.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(source_name="carolina", config=config)
        self.canonical_id: str = "carolina-c4ai/corpus-carolina"
        self.revision: str = "v2.0.1"
        self.pinned_commit_sha: str = CAROLINA_V2_0_1_COMMIT_SHA
        pop_cfg = self.config.get("population", {})
        if "taxonomies" in pop_cfg:
            self.population = dict(pop_cfg["taxonomies"])
        else:
            self.population = dict(TAXONOMY_POPULATION)

    def plan(
        self,
        mode: str = "representative",
        size: int = 10000,
        shards_per_taxonomy: int = 4,
        **kwargs,
    ) -> dict:
        """Generate a dry-run execution plan without performing transfers.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode.
        size : int, default 10000
            Target number of documents.
        shards_per_taxonomy : int, default 4
            Number of distributed shards to visit per taxonomy.
        **kwargs : any
            Additional arguments (ignored).

        Returns
        -------
        dict
            Dry-run execution plan.
        """
        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        taxonomies = list(TAXONOMY_DIRS.keys())

        if mode == "diagnostic":
            allocation = {
                "wik": int(size * 0.20),
                "dat": int(size * 0.20),
                "jud": int(size * 0.20),
                "leg": int(size * 0.20),
                "uni": int(size * 0.15),
                "soc": max(1, size - int(size * 0.95)),
            }
            frame = (
                f"stratified_quota_sampling (explicit quotas {allocation}; "
                f"up to {shards_per_taxonomy} distributed shards per taxonomy, "
                f"continuing until quota met or taxonomy exhausted)"
            )
        else:
            allocation = _proportional_allocation(size, self.population)
            frame = (
                f"proportional_population_sampling (weights from documented "
                f"taxonomy populations {self.population}; allocation "
                f"{allocation}; {shards_per_taxonomy} distributed shards per taxonomy)"
            )

        est_files = len(taxonomies) * shards_per_taxonomy
        est_mb = est_files * 4.0
        return {
            "source": self.source_name,
            "mode": mode,
            "target_size": size,
            "upstream_identifier": self.canonical_id,
            "upstream_revision": self.revision,
            "upstream_commit_sha": commit_sha,
            "population_scope": (
                "2,108,999 documents across 7 taxonomies in 823 xml.gz shards "
                "(Corpus Carolina v2.0.1, ~8.3 GB compressed)"
            ),
            "sampling_frame": frame,
            "taxonomy_allocation": allocation,
            "selected_partitions": [
                f"{t}: {shards_per_taxonomy} distributed shards (quota={allocation.get(t, '?')})"
                for t in taxonomies
            ],
            "safety_limits": {
                "shards_per_taxonomy": shards_per_taxonomy,
                "taxonomies_evaluated": len(taxonomies),
            },
            "estimated_transfer": f"~{est_mb:.1f} MB ({est_files} shards)",
        }

    def sample(
        self,
        mode: str = "representative",
        size: int = 10000,
        seed: int = 42,
        output_dir: Optional[Path | str] = None,
        shards_per_taxonomy: int = 4,
        **kwargs,
    ) -> Tuple[Path, ProvenanceManifest]:
        """Execute sampling on Corpus Carolina.

        Parameters
        ----------
        mode : str, default 'representative'
            Sampling mode ('representative' or 'diagnostic').
        size : int, default 10000
            Target number of documents.
        seed : int, default 42
            Deterministic seed.
        output_dir : Path or str or None, optional
            Output directory path.
        shards_per_taxonomy : int, default 4
            Number of distributed shards to visit per taxonomy.
        **kwargs : any
            Additional arguments (ignored).

        Returns
        -------
        tuple of (Path, ProvenanceManifest)
            Parquet file path and metadata manifest.
        """
        if output_dir is None:
            output_dir = Path("data/samples/gate_c1/carolina")
        else:
            output_dir = Path(output_dir)

        commit_sha = resolve_hf_commit_sha(self.canonical_id, revision="main")
        fs = HfFileSystem()
        byte_counter = [0]
        records_examined = 0
        stopping_reason = "shards_exhausted"

        if mode == "diagnostic":
            quotas = {
                "wik": int(size * 0.20),
                "dat": int(size * 0.20),
                "jud": int(size * 0.20),
                "leg": int(size * 0.20),
                "uni": int(size * 0.15),
                "soc": max(1, size - int(size * 0.95)),
            }
            sampler = StratifiedSampler(quotas=quotas, seed=seed)
            underfilled: Dict[str, int] = {}

            for taxonomy, quota in quotas.items():
                all_files = iter_carolina_xml_files(fs, taxonomy)
                # Stream ALL available shards until quota met or exhausted
                for raw_doc in stream_carolina_taxonomy(
                    fs,
                    taxonomy,
                    shard_indices=list(range(len(all_files))),
                    byte_counter=byte_counter,
                ):
                    records_examined += 1
                    key = raw_doc.get("original_id") or raw_doc["text"][:100]
                    norm_doc = validate_and_normalize(raw_doc)
                    sampler.add(taxonomy, key, norm_doc)
                    if len(sampler._reservoirs[taxonomy]) >= quota:
                        break

                actual = len(sampler._reservoirs[taxonomy])
                if actual < quota:
                    underfilled[taxonomy] = actual
                    logger.warning(
                        f"[carolina/diagnostic] Quota underfill: "
                        f"{taxonomy} obtained {actual}/{quota}"
                    )

            documents = sampler.get_all_samples()
            total_obtained = len(documents)
            if underfilled:
                stopping_reason = (
                    f"quota_underfill ({', '.join(f'{k}:{v}/{quotas[k]}' for k, v in underfilled.items())})"
                )
            elif total_obtained >= size:
                stopping_reason = "target_size_reached"

            sampling_frame = (
                f"stratified_quota_sampling (explicit quotas {quotas}; "
                f"all shards per underfilled taxonomy, stopping per-taxonomy "
                f"when quota met)"
            )

        else:
            # Representative mode: allocate proportionally, distribute shards
            allocation = _proportional_allocation(size, self.population)
            underfilled = {}
            all_documents: List[NormalizedDocument] = []

            for taxonomy, alloc in allocation.items():
                if alloc <= 0:
                    continue
                all_files = iter_carolina_xml_files(fs, taxonomy)
                if not all_files:
                    logger.warning(
                        f"[carolina/representative] No shards found for taxonomy '{taxonomy}'"
                    )
                    underfilled[taxonomy] = 0
                    continue

                # Select distributed shards across the full file list
                selected = _select_distributed_shards(
                    all_files, n_shards=shards_per_taxonomy, seed=seed
                )
                tax_reservoir = DeterministicReservoirSampler(
                    capacity=alloc, seed=seed
                )
                for raw_doc in stream_carolina_taxonomy(
                    fs,
                    taxonomy,
                    shard_indices=[all_files.index(s) for s in selected],
                    byte_counter=byte_counter,
                ):
                    records_examined += 1
                    key = raw_doc.get("original_id") or raw_doc["text"][:100]
                    norm_doc = validate_and_normalize(raw_doc)
                    tax_reservoir.add(key, norm_doc)
                    if len(tax_reservoir) >= alloc and tax_reservoir.total_seen >= alloc * 3:
                        break

                actual = len(tax_reservoir)
                if actual < alloc:
                    underfilled[taxonomy] = actual
                    logger.warning(
                        f"[carolina/representative] Allocation underfill: "
                        f"{taxonomy} obtained {actual}/{alloc}"
                    )
                all_documents.extend(tax_reservoir.get_sample())

            documents = all_documents
            total_obtained = len(documents)
            if underfilled:
                stopping_reason = (
                    f"allocation_underfill ({', '.join(f'{k}:{v}/{allocation[k]}' for k, v in underfilled.items())})"
                )
            elif total_obtained >= size:
                stopping_reason = "target_size_reached"

            sampling_frame = (
                f"proportional_population_sampling (taxonomy weights "
                f"{self.population}; per-taxonomy allocation {allocation}; "
                f"{shards_per_taxonomy} distributed shards per taxonomy)"
            )

        # Compute per-taxonomy actual counts for manifest
        taxonomy_counts: Dict[str, int] = {}
        for doc in documents:
            k = doc.subset or "unknown"
            taxonomy_counts[k] = taxonomy_counts.get(k, 0) + 1

        parquet_path, manifest = self._persist_sample(
            documents=documents,
            mode=mode,
            target_size=size,
            seed=seed,
            output_dir=output_dir,
            upstream_identifier=self.canonical_id,
            upstream_revision=self.revision,
            upstream_commit_sha=commit_sha,
            upstream_url=f"https://huggingface.co/datasets/{self.canonical_id}",
            upstream_configuration="corpus",
            population_scope=(
                "2,108,999 documents across 7 taxonomies in 823 xml.gz shards "
                "(Corpus Carolina v2.0.1, ~8.3 GB compressed)"
            ),
            sampling_frame=sampling_frame,
            records_examined=records_examined,
            bytes_read=byte_counter[0],
            stopping_reason=stopping_reason,
        )
        manifest.stats["taxonomy_allocation"] = (
            allocation if mode == "representative"
            else {k: v for k, v in quotas.items()}
        ) if mode in ("representative", "diagnostic") else {}
        manifest.stats["taxonomy_actual_counts"] = taxonomy_counts
        if underfilled:
            manifest.stats["underfill"] = underfilled
        manifest_path = output_dir / f"manifest_{mode}.json"
        manifest.save(manifest_path)
        return parquet_path, manifest
