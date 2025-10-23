"""
Evaluation Script

python test.py with mode=test \
snapshot=runs/PANet_ExpDisaster_align_1way_1shot_[train]/4/snapshots/30000.pth \
episode_specs_path=datasplits/disaster_1shot_splits.json

python test.py with mode=test snapshot=runs/PANet_ExpDisaster_align_1way_1shot_[train]/4/snapshots/30000.pth
"""
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import tqdm

from models.fewshot import FewShotSeg
from dataloaders.exp_disaster_fewshot import exp_disaster_fewshot
from util.metric import Metric
from util.utils import set_seed
from config import ex
from util.episode_utils import episode_specs_from_split, episode_specs_from_dicts


@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        if not _config['snapshot']:
            raise ValueError('Please provide a snapshot path when notrain is False.')
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    class_remap = _config['exp_disaster']['test']['class_remap']
    remap_values = [value for value in class_remap.values() if value not in {_config['ignore_label']}]
    remap_values.append(0)
    max_label = max(remap_values)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    label_list = []
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info('### Load data ###')
            dataset_kwargs = dict(
                images_dir=_config['path']['ExpDisaster']['meta_test_images'],
                labels_dir=_config['path']['ExpDisaster']['meta_test_labels'],
                class_remap=_config['exp_disaster']['test']['class_remap'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries'],
                max_iters=_config['n_steps'] * _config['batch_size'],
                ignore_label=_config['ignore_label'],
                allowed_classes=_config['exp_disaster']['test']['allowed_classes'],
                seed=_config['seed'] + run,
            )

            split_path = _config.get('episode_specs_path') or ''
            if split_path:
                episode_specs = episode_specs_from_split(
                    split_path,
                    n_ways=_config['task']['n_ways'],
                    n_shots=_config['task']['n_shots'],
                    n_queries=_config['task']['n_queries'],
                    class_remap=_config['exp_disaster']['test']['class_remap'],
                    ignore_label=_config['ignore_label'],
                )
                dataset_kwargs['episode_specs'] = episode_specs
                dataset_kwargs['max_iters'] = len(episode_specs)
            elif _config.get('episode_specs'):
                episode_specs = episode_specs_from_dicts(_config['episode_specs'])
                dataset_kwargs['episode_specs'] = episode_specs
                dataset_kwargs['max_iters'] = len(episode_specs)

            dataset = exp_disaster_fewshot(**dataset_kwargs)
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Episodes: {len(dataset)}")
            label_list = dataset.foreground_classes
            if not label_list:
                raise RuntimeError(
                    "No foreground classes available in constructed episodes; "
                    "check class_remap/allowed_classes configuration."
                )

            for sample_batched in tqdm.tqdm(testloader):
                label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)
                if query_labels.dim() == 4 and query_labels.size(1) == 1:
                    query_labels = query_labels.squeeze(1)

                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=label_list, n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=label_list)
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')
