"""Generate fixed support/query splits for the Exp_Disaster few-shot dataset.

This helper samples a deterministic support/query allocation per class and
persists it using the legacy ``tmp/val.json`` structure so downstream
pipelines can reuse the split without re-sampling episodes at runtime.


python -m script.generate_exp_disaster_split \
--images-dir ../_datasets/Exp_Disaster_Few-Shot/valset/images \
--labels-dir ../_datasets/Exp_Disaster_Few-Shot/valset/labels \
--output datasplits/exp_val_support_query.json \
--allowed-classes 1 2 \
--support-per-class 5 \
--query-per-class 100 \
--class-remap 0=0 20=1 30=2 \
--ignore-label 255 \
--seed 1234
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from dataloaders.exp_disaster_fewshot import ExpDisasterFewShotDataset


def _parse_kv_pairs(pairs: Sequence[str]) -> Dict[int, int]:
    """Parse ``KEY=VALUE`` command-line pairs into an integer mapping."""
    mapping: Dict[int, int] = {}
    for item in pairs:
        try:
            key_str, value_str = item.split("=", maxsplit=1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Could not parse '{item}'. Expected format 'KEY=VALUE'."
            ) from exc
        try:
            key = int(key_str)
            value = int(value_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Both key and value must be integers; received '{item}'."
            ) from exc
        mapping[key] = value
    return mapping


def _select_indices(
    candidates: Sequence[int],
    num_required: int,
    rng: random.Random,
    exclude: Iterable[int] | None = None,
) -> List[int]:
    """Select ``num_required`` indices, avoiding ``exclude`` when possible."""
    if not candidates:
        raise ValueError("No candidate samples supplied.")

    exclude_set = set(exclude or [])
    available = [idx for idx in candidates if idx not in exclude_set]

    picks: List[int] = []
    if len(available) >= num_required:
        picks.extend(rng.sample(available, num_required))
    else:
        picks.extend(available)
        while len(picks) < num_required:
            picks.append(rng.choice(candidates))
    return picks


def build_split(
    dataset: ExpDisasterFewShotDataset,
    *,
    class_ids: Sequence[int],
    support_per_class: int,
    query_per_class: int,
    seed: int,
) -> Dict[str, object]:
    """Construct deterministic support/query filename lists per class."""
    rng = random.Random(seed)
    support_set: Dict[str, List[str]] = {}
    query_set: Dict[str, List[str]] = {}

    for class_id in class_ids:
        if class_id not in dataset.class_to_indices:
            raise ValueError(
                f"Class id '{class_id}' is not present in the indexed dataset."
            )

        indices = dataset.class_to_indices[class_id]
        support_indices = _select_indices(indices, support_per_class, rng)
        query_indices = _select_indices(
            indices, query_per_class, rng, exclude=support_indices
        )

        support_files = [
            Path(dataset.samples[idx][0]).name for idx in support_indices
        ]
        query_files = [
            Path(dataset.samples[idx][0]).name for idx in query_indices
        ]

        support_set[str(class_id)] = support_files
        query_set[str(class_id)] = query_files

    return {"support_set": support_set, "query_set": query_set}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fixed Exp_Disaster support/query splits."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../_datasets/Exp_Disaster_Few-Shot/valset/images"),
        help="Directory containing Exp_Disaster validation RGB GeoTIFFs.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("../_datasets/Exp_Disaster_Few-Shot/valset/labels"),
        help="Directory containing Exp_Disaster validation label GeoTIFFs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasplits/exp_disaster_val_support_query.json"),
        help="Destination JSON path for the generated split.",
    )
    parser.add_argument(
        "--allowed-classes",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Remapped class ids to include in the split.",
    )
    parser.add_argument(
        "--support-per-class",
        type=int,
        default=5,
        help="Number of support tiles to sample per class.",
    )
    parser.add_argument(
        "--query-per-class",
        type=int,
        default=30,
        help="Number of query tiles to sample per class.",
    )
    parser.add_argument(
        "--class-remap",
        type=str,
        nargs="+",
        default=["0=0", "20=1", "30=2"],
        help="Mapping from raw labels to remapped ids (e.g. '20=1').",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Ignore label value used for nodata masking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    class_remap = _parse_kv_pairs(args.class_remap)
    allowed_classes = list(dict.fromkeys(args.allowed_classes))

    dataset = ExpDisasterFewShotDataset(
        images_dir=str(args.images_dir),
        labels_dir=str(args.labels_dir),
        class_remap=class_remap,
        n_ways=1,
        n_shots=args.support_per_class,
        n_queries=args.query_per_class,
        max_iters=1,
        ignore_label=args.ignore_label,
        allowed_classes=allowed_classes,
        seed=args.seed,
    )

    split = build_split(
        dataset,
        class_ids=allowed_classes,
        support_per_class=args.support_per_class,
        query_per_class=args.query_per_class,
        seed=args.seed,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    print(f"Saved Exp_Disaster split to {output_path.resolve()}")


if __name__ == "__main__":
    main()
