"""
Prediction visualization script for few-shot segmentation.
当前n_runs: 1，只运行一次。与 test 模式不同（test 通常 n_runs=5 计算均值和标准差）

Example:
python predict.py with mode=predict \
snapshot=runs/PANet_ExpDisaster_align_1way_1shot_[train]/3/snapshots/30000.pth \
episode_specs_path=datasplits/exp_val_support_query.json 
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import rasterio

from dataloaders.exp_disaster_fewshot import EpisodeSpec, exp_disaster_fewshot
from models.fewshot import FewShotSeg
from util.utils import set_seed
from util.visualize import (
    build_palette,
    mask_to_color,
    overlay_mask,
    save_geotiff,
    save_png,
    tensor_to_u8_image,
)
from config import ex
from util.episode_utils import episode_specs_from_split, episode_specs_from_dicts


def _persist_sources(_run) -> None:
    """Mirror source files into Sacred observer directory."""
    if not _run.observers:
        return
    observer = _run.observers[0]
    for source_file, _ in _run.experiment_info["sources"]:
        target = Path(observer.dir, "source", source_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        observer.save_file(source_file, str(target.relative_to(observer.dir)))
    shutil.rmtree(Path(observer.basedir) / "_sources", ignore_errors=True)


def _episode_spec_from_cli(config: Dict) -> Optional[List[EpisodeSpec]]:
    """Build a single episode specification from CLI overrides."""
    support_cli = config.get("support_list")
    if support_cli is None:
        return None
    if not isinstance(support_cli, dict):
        raise ValueError("`support_list` must be a mapping of class_id -> filenames.")
    query_cli = config.get("query_list")
    if not isinstance(query_cli, Iterable):
        raise ValueError("`query_list` must be an iterable of filenames.")

    class_ids = sorted(int(k) for k in support_cli.keys())
    support = {int(k): list(v) for k, v in support_cli.items()}
    query = list(query_cli)
    return [EpisodeSpec(class_ids=class_ids, support=support, query=query)]


def _inverse_remap(remap: Dict[int, int]) -> Dict[int, int]:
    """Return inverse mapping for ``class_remap`` dictionaries."""
    return {v: k for k, v in remap.items()}


def _coalesce_list(value):
    """Collapse DataLoader-batched singleton lists."""
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def _artifact_name(path: Path, run_root: Path) -> str:
    """Return a stable Sacred artifact name relative to the run root."""

    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return path.name


def _episode_metadata(sample: Dict, class_ids: List[int]) -> List[Dict[str, object]]:
    """Collect support metadata for JSON export."""
    support_images = _coalesce_list(sample.get("support_image_paths", []))
    support_labels = _coalesce_list(sample.get("support_label_paths", []))
    metadata: List[Dict[str, object]] = []
    for idx, class_id in enumerate(class_ids):
        metadata.append(
            {
                "class_id": int(class_id),
                "support_images": [str(path) for path in support_images[idx]],
                "support_labels": [str(path) for path in support_labels[idx]],
            }
        )
    return metadata


def _resolve_visual_dir(_config, _run, visualize_cfg) -> Path:
    """Determine the directory for visualization artefacts."""

    if _run.observers:
        base_dir = Path(_run.observers[0].dir)
    else:
        fallback = _config["path"].get("visual_dir") or os.getcwd()
        base_dir = Path(fallback)

    subdir = visualize_cfg.get("subdir")
    visual_dir = base_dir / subdir if subdir else base_dir
    visual_dir.mkdir(parents=True, exist_ok=True)
    return visual_dir


@ex.automain
def main(_run, _config, _log):
    _persist_sources(_run)

    set_seed(_config["seed"])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config["gpu_id"])
    torch.set_num_threads(1)

    _log.info("###### Create model ######")
    model = FewShotSeg(pretrained_path=_config["path"]["init_path"], cfg=_config["model"])
    model = nn.DataParallel(
        model.cuda(),
        device_ids=[
            _config["gpu_id"],
        ],
    )
    if not _config["notrain"] and _config.get("snapshot"):
        _log.info(f"Loading snapshot from {_config['snapshot']}")
        state_dict = torch.load(_config["snapshot"], map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()

    _log.info("###### Prepare data ######")
    visualize_cfg = _config.get("visualize", {})

    episode_specs: Optional[List[EpisodeSpec]] = None
    episode_specs_path = _config.get("episode_specs_path") or ""
    if episode_specs_path:
        episode_specs = episode_specs_from_split(
            episode_specs_path,
            n_ways=_config["task"]["n_ways"],
            n_shots=_config["task"]["n_shots"],
            n_queries=_config["task"]["n_queries"],
        )
    elif _config.get("episode_specs"):
        episode_specs = episode_specs_from_dicts(_config["episode_specs"])
    else:
        episode_specs = _episode_spec_from_cli(_config) or None

    dataset_kwargs = dict(
        images_dir=_config["path"]["ExpDisaster"]["meta_test_images"],
        labels_dir=_config["path"]["ExpDisaster"]["meta_test_labels"],
        class_remap=_config["exp_disaster"]["test"]["class_remap"],
        n_ways=_config["task"]["n_ways"],
        n_shots=_config["task"]["n_shots"],
        n_queries=_config["task"]["n_queries"],
        max_iters=_config["n_steps"] * _config["batch_size"],
        ignore_label=_config["ignore_label"],
        allowed_classes=_config["exp_disaster"]["test"]["allowed_classes"],
        seed=_config["seed"],
    )
    if episode_specs is not None:
        dataset_kwargs["episode_specs"] = episode_specs
        dataset_kwargs["max_iters"] = len(episode_specs)

    dataset = exp_disaster_fewshot(**dataset_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=_config["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    inverse_remap = _inverse_remap(_config["exp_disaster"]["test"]["class_remap"])
    visual_dir = _resolve_visual_dir(_config, _run, visualize_cfg)
    run_root = Path(_run.observers[0].dir) if _run.observers else visual_dir
    tif_dir = visual_dir / "tif_res"
    png_dir = visual_dir / "png_res"
    json_dir = visual_dir / "json_res"
    tif_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    remap_values = list(_config["exp_disaster"]["test"]["class_remap"].values())
    num_palette_classes = int(max(remap_values)) + 1 if remap_values else (_config["task"]["n_ways"] + 1)
    palette = build_palette(num_palette_classes, visualize_cfg.get("colormap", "default"))
    alpha = float(visualize_cfg.get("alpha", 0.6))
    overlay_mode = visualize_cfg.get("overlay_mode", "blend")
    if overlay_mode not in {"blend", "mask"}:
        raise ValueError("visualize.overlay_mode must be either 'blend' or 'mask', " f"got {overlay_mode!r}.")
    max_items = visualize_cfg.get("max_items")
    saved_items = 0

    _log.info("###### Predicting and visualizing ######")
    with torch.no_grad():
        for episode_idx, sample in enumerate(dataloader):
            support_images = [[shot.cuda() for shot in way] for way in sample["support_images"]]
            support_fg_mask = [[shot["fg_mask"].float().cuda() for shot in way] for way in sample["support_mask"]]
            support_bg_mask = [[shot["bg_mask"].float().cuda() for shot in way] for way in sample["support_mask"]]

            query_images = [query_image.cuda() for query_image in sample["query_images"]]
            query_images_raw = sample["query_images_t"]

            query_pred, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)
            pred_labels = query_pred.argmax(dim=1)

            episode_class_ids = [int(cid) for cid in sample["class_ids"]]
            label_mapping = {0: 0}
            for offset, class_id in enumerate(episode_class_ids, start=1):
                label_mapping[offset] = class_id

            support_meta = _episode_metadata(sample, episode_class_ids)
            query_image_paths_raw = _coalesce_list(sample["query_image_paths"])
            query_label_paths_raw = _coalesce_list(sample["query_label_paths"])
            query_image_paths = [Path(path) for path in query_image_paths_raw]
            query_label_paths = [Path(path) for path in query_label_paths_raw]

            for local_idx in range(pred_labels.shape[0]):
                if max_items is not None and saved_items >= max_items:
                    break

                pred_map = pred_labels[local_idx].detach().cpu().numpy().astype(np.uint8)
                remapped_pred = np.zeros_like(pred_map, dtype=np.uint8)
                for local_class, remapped_class in label_mapping.items():
                    remapped_pred[pred_map == local_class] = remapped_class

                original_pred = np.zeros_like(remapped_pred, dtype=np.uint8)
                for remapped_value, raw_value in inverse_remap.items():
                    original_pred[remapped_pred == remapped_value] = raw_value
                image_raw = tensor_to_u8_image(query_images_raw[local_idx])
                # Colorize using globally remapped labels so classes have consistent colors across episodes
                color_mask = mask_to_color(remapped_pred, palette, ignore_index=None)
                if overlay_mode == "mask":
                    overlay = color_mask
                else:
                    overlay = overlay_mask(image_raw, color_mask, alpha=alpha)

                stem = query_image_paths[local_idx].stem
                mask_path = tif_dir / f"{stem}_prediction.tif"
                overlay_path = png_dir / f"{stem}_overlay.png"
                metadata_path = json_dir / f"{stem}_meta.json"

                if visualize_cfg.get("save_mask", True):
                    with rasterio.open(query_label_paths[local_idx]) as src:
                        reference_profile = src.profile
                    save_geotiff(mask_path, original_pred.astype(np.uint8), reference_profile)
                    if _run.observers:
                        artifact_name = _artifact_name(mask_path, run_root)
                        _run.add_artifact(str(mask_path), name=artifact_name)

                if visualize_cfg.get("save_overlay", True):
                    save_png(overlay_path, overlay)
                    if _run.observers:
                        artifact_name = _artifact_name(overlay_path, run_root)
                        _run.add_artifact(str(overlay_path), name=artifact_name)

                if visualize_cfg.get("save_metadata", True):
                    metadata = {
                        "episode_index": episode_idx,
                        "query_index": local_idx,
                        "query_image": str(query_image_paths[local_idx]),
                        "query_label": str(query_label_paths[local_idx]),
                        "class_ids": episode_class_ids,
                        "label_mapping": label_mapping,
                        "support": support_meta,
                        "prediction_counts": {
                            str(label): int(count)
                            for label, count in zip(*np.unique(remapped_pred, return_counts=True))
                        },
                    }
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metadata_path, "w", encoding="utf-8") as handle:
                        json.dump(metadata, handle, indent=2)
                    if _run.observers:
                        artifact_name = _artifact_name(metadata_path, run_root)
                        _run.add_artifact(str(metadata_path), name=artifact_name)

                saved_items += 1
                _log.info(f"Saved prediction artefacts for {stem} to {visual_dir}")

            if max_items is not None and saved_items >= max_items:
                _log.info("Reached visualization limit; stopping early.")
                break

    _log.info(f"Finished saving {saved_items} prediction artefact(s) to {visual_dir}")
