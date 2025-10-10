"""Experiment configuration focused on the ExpDisaster dataset."""

import glob
import itertools
import os

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False
sacred.SETTINGS.CAPTURE_MODE = "no"

ex = Experiment("PANet")
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = [".", "./dataloaders", "./models", "./util", "./script"]
sources_to_save = list(itertools.chain.from_iterable([glob.glob(f"{folder}/*.py") for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    """Default Sacred configuration."""
    seed = 1234
    gpu_id = 0
    cuda_visable = "0"
    mode = "train"  # 'train', 'test', or 'predict'

    episode_specs_path = ""
    episode_specs = None

    model = {"align": True}
    task = {"n_ways": 1, "n_shots": 1, "n_queries": 1}
    ignore_label = 255

    path = {
        "log_dir": "./runs",
        "init_path": "./pretrained_model/vgg16-397923af.pth",
        "ExpDisaster": {
            "meta_train_images": "../_datasets/Exp_Disaster_Few-Shot/trainset/images",
            "meta_train_labels": "../_datasets/Exp_Disaster_Few-Shot/trainset/labels",
            "meta_test_images": "../_datasets/Exp_Disaster_Few-Shot/valset/images",
            "meta_test_labels": "../_datasets/Exp_Disaster_Few-Shot/valset/labels",
        },
    }

    exp_disaster = {
        "train": {
            "class_remap": {i: i for i in range(9)},
            "allowed_classes": [1, 2, 3, 4, 5, 6, 7, 8],
        },
        "test": {
            "class_remap": {0: 0, 20: 1, 30: 2},
            "allowed_classes": [1, 2],
        },
    }

    if mode == "train":
        n_steps = 30000
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        print_interval = 100
        save_pred_every = 10000
        optim = {"lr": 1e-3, "momentum": 0.9, "weight_decay": 0.0005}

    elif mode == "test":
        notrain = False
        snapshot = ""
        n_runs = 5
        n_steps = 100
        batch_size = 1

    elif mode == "predict":
        notrain = False
        snapshot = ""
        n_runs = 1
        n_steps = 100
        batch_size = 1
        visualize = {
            "save_mask": True,
            "save_overlay": True,
            "save_metadata": True,
            "alpha": 0.6,
            "colormap": "exp_disaster_binary",
            "overlay_mode": "mask",
            "subdir": "",
            "max_items": None,
        }

    else:
        raise ValueError('Unsupported "mode" provided.')

    exp_components = ["ExpDisaster"]
    exp_components.extend([key for key, value in model.items() if value])
    exp_components.append(f"{task['n_ways']}way_{task['n_shots']}shot_[{mode}]")
    exp_str = "_".join(exp_components)


@ex.config_hook
def add_observer(config, command_name, logger):
    """Attach a FileStorageObserver so Sacred logs and artefacts persist."""
    exp_name = f"{ex.path}_{config['exp_str']}"
    run_dir = os.path.join(config["path"]["log_dir"], exp_name)
    observer = FileStorageObserver.create(run_dir)
    ex.observers.append(observer)
    config["path"]["observer_dir"] = run_dir
    if config["mode"] == "predict":
        visualize_cfg = config.get("visualize") or {}
        subdir = visualize_cfg.get("subdir")
        visualize_cfg.setdefault("subdir", subdir or "")
        visualize_cfg.setdefault("save_mask", True)
        visualize_cfg.setdefault("save_overlay", True)
        visualize_cfg.setdefault("save_metadata", True)
        visualize_cfg.setdefault("alpha", 0.6)
        visualize_cfg.setdefault("colormap", "default")
        visualize_cfg.setdefault("overlay_mode", "blend")
        visualize_cfg.setdefault("max_items", None)
        config["visualize"] = visualize_cfg
        config["path"]["visual_dir"] = os.path.join(run_dir, subdir) if subdir else run_dir
    return config
