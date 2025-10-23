# PANet for ExpDisaster Few-Shot Segmentation

PANet implements prototype alignment few-shot semantic segmentation. This repository tailors the ICCV'19 codebase to the ExpDisaster dataset with Sacred-based experiment tracking and extended visualization utilities.

## Environment Setup
1. Create a Python 3.8+ environment and install the dependencies listed in `requirements.txt` or the original paper (PyTorch ≥1.0.1, torchvision ≥0.2.1, sacred, rasterio, tqdm, NumPy, SciPy, PIL, pycocotools).
2. Download ImageNet-pretrained VGG16 weights from `https://download.pytorch.org/models/vgg16-397923af.pth` and place the file at `pretrained_model/vgg16-397923af.pth`.
3. Ensure dataset symlinks or folders follow:
   - `../_datasets/Exp_Disaster_Few-Shot/trainset/{images,labels}`
   - `../_datasets/Exp_Disaster_Few-Shot/valset/{images,labels}`

## Core Files
- `config.py`: Sacred experiment knobs; set `mode ∈ {train,test,predict}`. Defines paths, task configuration, logging directories and visualization defaults.
- `train.py`: episodic meta-training entry point.
- `test.py`: evaluation script producing mIoU metrics over validation episodes.
- `predict.py`: deterministic prediction & visualization pipeline that saves GeoTIFF masks, PNGs and metadata JSON.
- `dataloaders/exp_disaster_fewshot.py`: episode sampling logic for ExpDisaster tiles (GeoTIFF IO, class remapping, deterministic episode specs).
- `util/visualize.py`: tensor ↔ image helpers, color palettes (including `exp_disaster_binary`: background black, landslide red, flood blue), overlay utilities.

## Experiment Workflow
### 1. Configure Sacred
```
python train.py with mode=train gpu_id=0
```
- Edit `config.py` or override CLI parameters (`task.n_ways`, `task.n_shots`, `exp_disaster.train.allowed_classes`, etc.).
- Sacred stores runs under `runs/PANet_<exp_str>/` with source snapshotting.

### 2. Train PANet on ExpDisaster
```
python train.py with mode=train \
    task.n_ways=1 task.n_shots=1 task.n_queries=1 \
    n_steps=30000 gpu_id=0
```
- Each iteration samples an episode, computes prototype-aligned loss, and checkpoints to `runs/.../snapshots/` according to `save_pred_every`.
- Training relies on `exp_disaster_fewshot` to remap landcover classes `{1..8}` and ignore nodata pixels.

### 3. Generate Deterministic Episode JSON (Optional)
```
python script/generate_disaster_splits.py \
    --path ../_datasets/Exp_Disaster_Few-Shot \
    --shots 1
```
- Produces `datasplits/disaster_1shot_splits.json` with:
  - `support: {images: [...], labels: [...]}`
  - `query:   {images: [...], labels: [...]}`
- The parser infers per-class membership from labels using `exp_disaster.test.class_remap`.

### 4. Evaluate on Validation Episodes
```
python test.py with mode=test \
    snapshot=runs/PANet_ExpDisaster_align_1way_1shot_[train]/<run>/snapshots/30000.pth \
    episode_specs_path=datasplits/disaster_1shot_splits.json \
    gpu_id=0
```
- `test.py` runs `n_runs` repetitions, logs per-class IoU, binary IoU, means and stds to Sacred.
- Without `episode_specs_path`, it samples random episodes respecting `allowed_classes` (default `{1,2}` which map to landslide/flood in validation).

### 5. Predict & Visualize
```
python predict.py with mode=predict \
    snapshot=runs/PANet_ExpDisaster_align_1way_1shot_[train]/<run>/snapshots/30000.pth \
    episode_specs_path=datasplits/disaster_1shot_splits.json \
    visualize.subdir=PANet_ExpDisaster_align_1way_1shot_[predict]
```
- Uses `visualize.colormap="exp_disaster_binary"` and `visualize.overlay_mode="mask"` by default to render PNG masks (black background, landslide red, flood blue).
- Output structure per run: `runs/.../tif_res`, `png_res`, `json_res`.
- Override visualization preferences via CLI, e.g. `visualize.overlay_mode=blend visualize.alpha=0.5` for overlayed imagery.

### 6. Inspect Artefacts and Metrics
- GeoTIFF masks preserve geospatial metadata for GIS workflows.
- PNG masks and overlays live under `runs/<predict_exp>/png_res/`.
- JSON metadata lists support/query filenames, class IDs, label remaps and per-mask pixel counts.
- Sacred observers automatically register these files as artefacts for reproducibility.

## Tips
- Set `episode_specs_path` or inline `support_list`, `query_list` CLI overrides to reproduce deterministic experiments.
- Adjust `visualize.max_items` to subsample predictions without rerunning the model.
- Ensure `torch.cuda.set_device` matches the desired GPU index (`gpu_id`).
- For quick smoke tests, lower `n_steps` or `n_runs` in configs to verify data pipelines before long training runs.

## Citation
If PANet benefits your research, please cite Wang et al., ICCV 2019.
