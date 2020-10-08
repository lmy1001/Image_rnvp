import os
from time import time
from sys import stdout
from scipy.stats import entropy

import h5py as h5
import numpy as np
import torch

def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    saving_mode = kwargs.get('saving_mode')
    if saving_mode:
        print('save results\n')
        clouds_fname = '{}_{}_{}_{}_clouds_{}.h5'.format(kwargs['model_name'][:-4],
                                                         iterator.dataset.part,
                                                         kwargs['cloud_size'],
                                                         kwargs['sampled_cloud_size'],
                                                         kwargs['usage_mode'])
        clouds_fname = os.path.join(kwargs['path2save'], clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_samples'] * len(iterator.dataset), 3,
                   kwargs['cloud_size'] if kwargs['refine_sampled_clouds'] else kwargs['sampled_cloud_size']),
            dtype=np.float32
        )
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_samples'] * len(iterator.dataset), 3,
                   kwargs['cloud_size']),
            dtype=np.float32
        )

    model.eval()
    torch.set_grad_enabled(False)

    for i, batch in enumerate(iterator):
        clouds = batch['cloud'].cuda(non_blocking=True)
        ref_clouds = batch['eval_cloud'].cuda(non_blocking=True)
        if 'ic' in train_mode:
            images = batch['image'].cuda(non_blocking=True)
            outputs = model(clouds, images, n_sampled_points=kwargs['sampled_cloud_size'])
        else:
            outputs = model(clouds, n_sampled_points=kwargs['sampled_cloud_size'])


        if kwargs.get('usage_mode') == 'predicting':
            rec_clouds = outputs['p_prior_samples'][-1]

            if kwargs['unit_scale_evaluation']:
                if kwargs['cloud_scale']:
                    rec_clouds *= kwargs['cloud_scale_scale']
                    ref_clouds *= kwargs['cloud_scale_scale']

            if saving_mode:
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + rec_clouds.shape[0]] = \
                    rec_clouds.cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + ref_clouds.shape[0]] = \
                    ref_clouds.cpu().numpy().astype(np.float32)

            rec_clouds = torch.transpose(rec_clouds, 1, 2)
            ref_clouds = torch.transpose(ref_clouds, 1, 2)

    if saving_mode:
        clouds_file.close()
