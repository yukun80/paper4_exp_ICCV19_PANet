# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py` and `test.py`, driven by Sacred configs in `config.py`. Backbone and head code stays in `models/`, while the disaster-specific sampler lives in `dataloaders/exp_disaster_fewshot.py`. Shared helpers sit in `util/`, and `experiments/` holds reproducible shell launchers. Store pretrained weights in `pretrained_model/` and expect Sacred to log artifacts under `runs/`.

## Build, Test, and Development Commands
- `python train.py with dataset=ExpDisaster mode=train task.n_ways=1 task.n_shots=5 patch_size=512 gpu_id=0`: episodic meta-training on landcover tiles.
- `python test.py with dataset=ExpDisaster mode=test support_list=['sample_01.tif'] query_list=['sample_02.tif'] gpu_id=0`: adapt on valset support patches and score the paired query set.
- `python train.py with dataset=VOC ...` or `bash experiments/voc_1way_1shot_[train].sh`: maintain legacy VOC/COCO baselines for regression.
Ensure `config.py:path['ExpDisaster']` points at `../_datasets/Exp_Disaster_Few-Shot` and that `pretrained_model/vgg16-397923af.pth` is available before launching jobs.

## ExpDisaster Prediction Visuals
- Extend Sacred with `mode='predict'`: reuse `exp_disaster` config blocks, force `batch_size=1`, `n_runs=1`, expose a `visualize` dict (`save_mask`, `save_overlay`, `alpha`, `colormap`), and append `[predict]` to `exp_str`.
- Add `predict.py` patterned after `test.py`: seed setup, CUDA binding, instantiate `FewShotSeg`, optionally load `snapshot`, and build a `DataLoader` over `exp_disaster_fewshot`.
- Support deterministic episodes via `episode_specs` (JSON/YAML) or CLI `support_list`/`query_list` routed into the dataset; keep `allowed_classes` filtering and `ignore_label` handling consistent.
- For each query, `argmax` logits, map indices back through the inverse `class_remap`, and save artefacts under `runs/<exp>/visuals`: GeoTIFF mask with source profile, PNG overlay blending `query_images_t` and a palette-colored mask, plus optional JSON metadata (support filenames, class IDs, remap used).
- Factor tensor→image and overlay utilities into `util/visualize.py` (registered via `ex.add_source_file`); log every saved file via `_log.info` and `_run.add_artifact` so Sacred observers capture visual outputs.
- Usage example: `python predict.py with mode=predict dataset=ExpDisaster snapshot=/path/to/model.pth visualize.max_items=10`. Deterministic episodes: supply `episode_specs_path=specs.json` (list of `{"class_ids": [...], "support": {"1": [...], ...}, "query": [...]}`) or inline `support_list={'1': ['tile_a.tif']} query_list=['tile_b.tif']`.

## Exp_Disaster Few-Shot Workflow
`../_datasets/Exp_Disaster_Few-Shot/trainset/{images,labels}` and `valset/{images,labels}` contain paired 512×512 RGB GeoTIFFs. Train labels encode 0–8, while val labels use {0,20,30} with 0 as background. The loader flattens these directories, remaps landcover classes to {0…8} and disaster classes to {0,1,2}, and masks nodata pixels using raster metadata. Episodes sample support/query patches by scanning each label raster for the requested class; pass explicit filename lists when you need deterministic splits.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation and ≤120-character lines. Keep modules, functions, and Sacred keys in `snake_case`, classes in `CamelCase`, and mirror the concise docstrings used in `models/fewshot.py`. Prefer vectorized tensor ops, reuse helpers in `util/`, and register new modules with `ex.add_source_file` so Sacred captures dependencies.

## Testing Guidelines
There are no unit tests; rely on Sacred runs. For ExpDisaster jobs, log the class-remap dictionary, support/query filenames, and per-class mIoU into `runs/<exp>/metrics.json`. Keep new experiment scripts aligned with the `dataset_[variants]_[mode].sh` naming. Use short smoke runs (`n_steps`, `n_runs` small) to validate the flat-directory loader before committing longer experiments.

## Commit & Pull Request Guidelines
Commits in history are short and lower case, so use concise imperative subjects such as `add exp disaster loader`. Group related files per commit and call out any Sacred config keys you changed. In pull requests, summarize the goal, key files, dataset prerequisites (especially ExpDisaster paths), and include the command plus metric snapshot used for verification. Link the relevant `runs/` directory and mention any data-preparation steps reviewers must reproduce.
