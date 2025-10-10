"""Exp_Disaster_Few-Shot episodic dataset utilities for PANet."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# ImageNet statistics keep parity with the pre-trained backbones.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass(frozen=True)
class EpisodeSpec:
    """Deterministic description of a few-shot episode."""

    class_ids: Sequence[int]
    support: Dict[int, Sequence[str]]
    query: Sequence[str]


class ExpDisasterFewShotDataset(Dataset):
    """Episode generator for the Exp_Disaster_Few-Shot dataset.

    The dataset layout is expected to be::

        root/
            trainset/
                images/*.tif
                labels/*.tif
            valset/
                images/*.tif
                labels/*.tif

    Each tile is a 512x512 RGB GeoTIFF paired with a single-channel label
    GeoTIFF. Landcover tiles (trainset) encode classes in the range 0–8, while
    disaster tiles (valset) use {0, 20, 30}. Both splits may contain nodata
    values which are remapped to ``ignore_label``.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        class_remap: Dict[int, int],
        *,
        n_ways: int,
        n_shots: int,
        n_queries: int,
        max_iters: int,
        ignore_label: int = 255,
        allowed_classes: Optional[Iterable[int]] = None,
        episode_specs: Optional[Sequence[EpisodeSpec]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_remap = class_remap
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.max_iters = max_iters if episode_specs is None else len(episode_specs)
        self.ignore_label = ignore_label
        self.random_state = random.Random(seed)
        self.episode_specs = list(episode_specs) if episode_specs is not None else None

        self._remap_lut = self._build_remap_lut()
        (
            self.samples,
            self.class_to_indices,
            self.stem_to_index,
            self.index_to_classes,
        ) = self._index_dataset(allowed_classes)

        if not self.samples:
            raise RuntimeError(
                f"No image/label pairs found under '{images_dir}' and '{labels_dir}'."
            )

        if self.episode_specs is None:
            available_classes = sorted(self.class_to_indices.keys())
            if allowed_classes is not None:
                allowed = set(allowed_classes)
                available_classes = [cid for cid in available_classes if cid in allowed]
            if not available_classes:
                raise ValueError(
                    "No eligible foreground classes remain after indexing the dataset."
                )
            self.episode_classes = available_classes
        else:
            missing = [
                spec for spec in self.episode_specs
                if not set(spec.class_ids).issubset(self.class_to_indices.keys())
            ]
            if missing:
                raise ValueError(
                    "Episode specs reference classes that are not available in the dataset: "
                    f"{missing}"
                )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.max_iters

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.episode_specs is not None:
            spec = self.episode_specs[index]
            return self._build_episode_from_spec(spec)
        return self._build_random_episode()

    # ------------------------------------------------------------------
    # Episode builders
    # ------------------------------------------------------------------

    def _build_random_episode(self) -> Dict[str, object]:
        class_ids = self._sample_classes()
        support_bins = []
        query_bins = []
        for class_id in class_ids:
            support_indices, query_indices = self._sample_indices_for_class(class_id)
            support_bins.append(support_indices)
            query_bins.append(query_indices)
        return self._assemble_episode(class_ids, support_bins, query_bins)

    def _build_episode_from_spec(self, spec: EpisodeSpec) -> Dict[str, object]:
        class_ids = list(spec.class_ids)
        if len(class_ids) != self.n_ways:
            raise ValueError(
                f"Episode spec defines {len(class_ids)} classes but n_ways={self.n_ways}."
            )

        support_bins: List[List[int]] = []
        for class_id in class_ids:
            filenames = list(spec.support.get(class_id, []))
            if len(filenames) < self.n_shots:
                raise ValueError(
                    f"Episode spec provides {len(filenames)} support samples for class"
                    f" {class_id}, fewer than n_shots={self.n_shots}."
                )
            indices = [self._filename_to_index(name) for name in filenames[: self.n_shots]]
            support_bins.append(indices)

        required_queries = self.n_ways * self.n_queries
        if len(spec.query) < required_queries:
            raise ValueError(
                f"Episode spec lists {len(spec.query)} query samples but "
                f"requires at least {required_queries}."
            )

        remaining = {class_id: self.n_queries for class_id in class_ids}
        query_bins = [[] for _ in class_ids]
        for filename in spec.query:
            if all(count == 0 for count in remaining.values()):
                break
            idx = self._filename_to_index(filename)
            present = self.index_to_classes[idx] & set(class_ids)
            if not present:
                continue
            for class_id in present:
                pos = class_ids.index(class_id)
                if remaining[class_id] > 0:
                    query_bins[pos].append(idx)
                    remaining[class_id] -= 1
                    break

        exhausted = [cid for cid, count in remaining.items() if count > 0]
        if exhausted:
            raise ValueError(
                "Episode spec does not provide enough query samples for classes: "
                f"{exhausted}"
            )

        return self._assemble_episode(class_ids, support_bins, query_bins)

    def _assemble_episode(
        self,
        class_ids: Sequence[int],
        support_bins: Sequence[Sequence[int]],
        query_bins: Sequence[Sequence[int]],
    ) -> Dict[str, object]:
        support_images: List[List[torch.Tensor]] = []
        support_images_t: List[List[torch.Tensor]] = []
        support_mask: List[List[Dict[str, torch.Tensor]]] = []
        support_inst: List[List[torch.Tensor]] = []

        query_images: List[torch.Tensor] = []
        query_images_t: List[torch.Tensor] = []
        query_labels: List[torch.Tensor] = []
        query_masks: List[List[torch.Tensor]] = []
        query_cls_idx: List[List[int]] = []
        support_image_paths: List[List[str]] = []
        support_label_paths: List[List[str]] = []
        query_image_paths: List[str] = []
        query_label_paths: List[str] = []

        for class_id, support_indices in zip(class_ids, support_bins):
            support_images.append([])
            support_images_t.append([])
            support_mask.append([])
            support_inst.append([])
            support_image_paths.append([])
            support_label_paths.append([])
            for idx in support_indices:
                image_norm, image_raw, label = self._load_pair(idx)
                support_images[-1].append(image_norm)
                support_images_t[-1].append(image_raw)
                support_mask[-1].append(
                    self._make_support_masks(label, class_id, class_ids)
                )
                support_inst[-1].append(torch.zeros_like(label, dtype=torch.long))
                image_path, label_path = self.samples[idx]
                support_image_paths[-1].append(image_path)
                support_label_paths[-1].append(label_path)

        for class_id, query_indices in zip(class_ids, query_bins):
            if len(query_indices) < self.n_queries:
                raise ValueError(
                    f"Class {class_id} only has {len(query_indices)} query samples; "
                    f"need {self.n_queries}."
                )
            for idx in query_indices[: self.n_queries]:
                image_norm, image_raw, label = self._load_pair(idx)
                tgt_label, masks, cls_idx = self._make_query_targets(label, class_ids)
                query_images.append(image_norm)
                query_images_t.append(image_raw)
                query_labels.append(tgt_label)
                query_masks.append(masks)
                query_cls_idx.append(cls_idx)
                image_path, label_path = self.samples[idx]
                query_image_paths.append(image_path)
                query_label_paths.append(label_path)

        return {
            'class_ids': list(class_ids),
            'support_images': support_images,
            'support_images_t': support_images_t,
            'support_mask': support_mask,
            'support_inst': support_inst,
            'query_images': query_images,
            'query_images_t': query_images_t,
            'query_labels': query_labels,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
            'support_image_paths': support_image_paths,
            'support_label_paths': support_label_paths,
            'query_image_paths': query_image_paths,
            'query_label_paths': query_label_paths,
        }

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------

    def _sample_classes(self) -> List[int]:
        if len(self.episode_classes) < self.n_ways:
            raise ValueError(
                f"Requested n_ways={self.n_ways} but only "
                f"{len(self.episode_classes)} classes available."
            )
        return self.random_state.sample(self.episode_classes, self.n_ways)

    def _sample_indices_for_class(self, class_id: int) -> Tuple[List[int], List[int]]:
        candidates = self.class_to_indices[class_id]
        if not candidates:
            raise ValueError(f"No samples available for class {class_id}.")

        required = self.n_shots + self.n_queries
        if len(candidates) >= required:
            sampled = self.random_state.sample(candidates, required)
        else:
            sampled = [self.random_state.choice(candidates) for _ in range(required)]

        support = sampled[: self.n_shots]
        query = sampled[self.n_shots :]
        if len(query) < self.n_queries:
            query.extend([sampled[-1]] * (self.n_queries - len(query)))
        return support, query

    # ------------------------------------------------------------------
    # Data IO helpers
    # ------------------------------------------------------------------

    def _index_dataset(
        self,
        allowed_classes: Optional[Iterable[int]],
    ) -> Tuple[
        List[Tuple[str, str]],
        Dict[int, List[int]],
        Dict[str, int],
        Dict[int, set],
    ]:
        image_files = sorted(
            f for f in os.listdir(self.images_dir) if f.lower().endswith('.tif')
        )
        label_files = sorted(
            f for f in os.listdir(self.labels_dir) if f.lower().endswith('.tif')
        )
        label_lookup = {os.path.splitext(f)[0]: f for f in label_files}

        samples: List[Tuple[str, str]] = []
        class_to_indices: Dict[int, List[int]] = defaultdict(list)
        stem_to_index: Dict[str, int] = {}
        index_to_classes: Dict[int, set] = {}

        allowed_set = set(allowed_classes) if allowed_classes is not None else None

        for image_file in image_files:
            stem = os.path.splitext(image_file)[0]
            if stem not in label_lookup:
                continue
            label_path = os.path.join(self.labels_dir, label_lookup[stem])
            remapped = self._read_label(label_path)
            sample_classes = set(np.unique(remapped).tolist())
            sample_classes.discard(0)
            sample_classes.discard(self.ignore_label)
            if allowed_set is not None:
                sample_classes &= allowed_set

            sample_idx = len(samples)
            samples.append((os.path.join(self.images_dir, image_file), label_path))
            stem_to_index[stem] = sample_idx
            index_to_classes[sample_idx] = sample_classes

            for class_id in sample_classes:
                class_to_indices[class_id].append(sample_idx)

        return samples, class_to_indices, stem_to_index, index_to_classes

    def _read_label(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            label = src.read(1)
            nodata = src.nodata
        remapped = self._remap_lut[label]
        if nodata is not None:
            remapped[label == nodata] = self.ignore_label
        return remapped

    def _load_pair(self, sample_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[sample_index]
        image, image_raw = self._read_image(image_path)
        label = torch.from_numpy(self._read_label(label_path)).long()
        return image, image_raw, label

    def _filename_to_index(self, filename: str) -> int:
        stem = os.path.splitext(os.path.basename(filename))[0]
        if stem not in self.stem_to_index:
            raise KeyError(f"Filename '{filename}' not found in indexed samples.")
        return self.stem_to_index[stem]

    def _read_image(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
        if image.shape[0] != 3:
            raise ValueError(
                f"Expected 3-channel RGB tile, found {image.shape[0]} channels at '{path}'."
            )
        image_raw = torch.from_numpy(image.copy())
        image_norm = (image_raw / 255.0 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return image_norm, image_raw

    def _make_support_masks(
        self,
        label: torch.Tensor,
        target_class: int,
        episode_classes: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        fg = (label == target_class).float()
        bg = torch.ones_like(fg)
        bg[label == self.ignore_label] = 0
        for class_id in episode_classes:
            bg[label == class_id] = 0
        fg_scribble = fg.clone().long()
        bg_scribble = bg.clone().long()
        return {
            'fg_mask': fg,
            'bg_mask': bg,
            'fg_scribble': fg_scribble,
            'bg_scribble': bg_scribble,
        }

    def _make_query_targets(
        self,
        label: torch.Tensor,
        episode_classes: Sequence[int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
        query_label = torch.zeros((1, *label.shape), dtype=torch.long)
        query_label[0][label == self.ignore_label] = self.ignore_label

        masks = [((label == 0) & (label != self.ignore_label)).float().unsqueeze(0)]
        cls_idx = [0]

        for idx, class_id in enumerate(episode_classes, start=1):
            class_mask = label == class_id
            if class_mask.any():
                query_label[0][class_mask] = idx
                masks.append(class_mask.float().unsqueeze(0))
                cls_idx.append(idx)

        return query_label, masks, cls_idx

    def _build_remap_lut(self) -> np.ndarray:
        lut = np.full(256, self.ignore_label, dtype=np.uint8)
        for raw_value, mapped_value in self.class_remap.items():
            if raw_value < 0 or raw_value >= len(lut):
                raise ValueError(
                    f"Raw label value {raw_value} exceeds the LUT range [0, 255]."
                )
            lut[raw_value] = mapped_value
        return lut


def exp_disaster_fewshot(
    images_dir: str,
    labels_dir: str,
    class_remap: Dict[int, int],
    *,
    n_ways: int,
    n_shots: int,
    n_queries: int,
    max_iters: int,
    ignore_label: int = 255,
    allowed_classes: Optional[Iterable[int]] = None,
    episode_specs: Optional[Sequence[EpisodeSpec]] = None,
    seed: Optional[int] = None,
) -> ExpDisasterFewShotDataset:
    """Factory mirroring the VOC/COCO helpers for parity with train/test scripts."""

    return ExpDisasterFewShotDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_remap=class_remap,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=n_queries,
        max_iters=max_iters,
        ignore_label=ignore_label,
        allowed_classes=allowed_classes,
        episode_specs=episode_specs,
        seed=seed,
    )
