"""Utilities for constructing deterministic ExpDisaster episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dataloaders.exp_disaster_fewshot import EpisodeSpec


def _ensure_int_keys(mapping: Dict[str, Iterable[str]]) -> Dict[int, List[str]]:
    """Convert stringified integer keys to integer keys."""
    result: Dict[int, List[str]] = {}
    for key, values in mapping.items():
        try:
            class_id = int(key)
        except ValueError as exc:
            raise ValueError(f"Expected integer-like key, received '{key}'.") from exc
        result[class_id] = list(values)
    return result


def _chunk(sequence: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield successive chunks from ``sequence`` with length ``size``."""
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for idx in range(0, len(sequence), size):
        chunk = list(sequence[idx : idx + size])
        if len(chunk) < size:
            break
        yield chunk


def episode_specs_from_split(
    split_path: str | Path,
    *,
    n_ways: int,
    n_shots: int,
    n_queries: int,
) -> List[EpisodeSpec]:
    """Convert a support/query JSON split into deterministic ``EpisodeSpec``s.

    The expected JSON structure matches the output of
    ``util/generate_exp_disaster_split.py``::

        {
            "support_set": {"1": [...], "2": [...]},
            "query_set": {"1": [...], "2": [...]},
            "meta": {...}
        }

    Returns
    -------
    List[EpisodeSpec]
        Deterministic episode specifications compatible with the dataset.
    """
    path = Path(split_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        # Backwards compatibility: list of EpisodeSpec-like dicts.
        specs = []
        for entry in data:
            specs.append(
                EpisodeSpec(
                    class_ids=entry["class_ids"],
                    support={int(k): v for k, v in entry["support"].items()},
                    query=entry["query"],
                )
            )
        return specs

    if not isinstance(data, dict):
        raise ValueError("Unsupported split file format.")

    if "support_set" not in data or "query_set" not in data:
        raise ValueError("Split JSON must contain 'support_set' and 'query_set'.")

    support_set = _ensure_int_keys(data["support_set"])
    raw_query_set = data["query_set"]
    if isinstance(raw_query_set, list):
        query_set_map = {class_id: list(raw_query_set) for class_id in support_set.keys()}
    else:
        query_set_map = _ensure_int_keys(raw_query_set)

    classes = sorted(support_set.keys())
    if n_ways != 1:
        raise NotImplementedError(
            "episode_specs_from_split currently supports n_ways=1 only."
        )

    episode_specs: List[EpisodeSpec] = []
    for class_id in classes:
        support_files = support_set[class_id]
        query_files = query_set_map.get(class_id, [])

        if len(support_files) < n_shots:
            raise ValueError(
                f"Class {class_id} only lists {len(support_files)} support samples; "
                f"need at least {n_shots}."
            )

        support_subset = support_files[:n_shots]
        for query_chunk in _chunk(query_files, n_queries):
            episode_specs.append(
                EpisodeSpec(
                    class_ids=[class_id],
                    support={class_id: support_subset},
                    query=query_chunk,
                )
            )

    if not episode_specs:
        raise ValueError(
            "No episode specifications generated; check query_set contents."
        )

    return episode_specs


def episode_specs_from_dicts(entries: Sequence[Dict]) -> List[EpisodeSpec]:
    """Convert a list of dictionaries into concrete ``EpisodeSpec`` objects."""
    episode_specs: List[EpisodeSpec] = []
    for entry in entries:
        if "class_ids" not in entry:
            raise ValueError("Each episode spec must contain 'class_ids'.")
        episode_specs.append(
            EpisodeSpec(
                class_ids=entry["class_ids"],
                support={int(k): v for k, v in entry.get("support", {}).items()},
                query=entry.get("query", []),
            )
        )
    return episode_specs
