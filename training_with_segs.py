import os
from time import time
from sys import stdout
import torch
import h5py as h5
import numpy as np
import torch.nn as nn
from networks.models import Conditional_RNVP_with_image_prior
from networks.losses import Conditional_RNVP_with_image_prior_loss
from networks.optimizers import Adam, LRUpdater

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(state, model_name):
    torch.save(state, model_name)
    print('Model saved to ' + model_name)

def cnt_params(params):
    return sum(p.numel() for p in params if p.requires_grad)


def save_point_clouds(batch_i, gt_cloud, gen_cloud, image, len_dataset, **kwargs):

    clouds_fname = '{}_{}_{}_{}_segs_clouds.h5'.format(kwargs['model_name'][:-4],
                                                     kwargs['cloud_size'],
                                                     kwargs['sampled_cloud_size'],
                                                     kwargs['usage_mode'])
    cloud_fname = os.path.join(kwargs['path2save'], clouds_fname)
    if not os.path.exists(cloud_fname):
        clouds_file = h5.File(cloud_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_samples'] * len_dataset, 3,
                   kwargs['cloud_size']),dtype=np.float32)
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_samples'] * len_dataset, 3,
                   kwargs['cloud_size']),dtype=np.float32)
        gt_images = clouds_file.create_dataset(
            'images',
            shape=(kwargs['N_samples'] * len_dataset, 3, 137, 137), dtype=np.uint8)
    else:
        clouds_file = h5.File(cloud_fname, 'a')
        sampled_clouds = clouds_file['sampled_clouds']
        gt_clouds = clouds_file['gt_clouds']
        gt_images = clouds_file['images']

    gen = torch.zeros((kwargs['batch_size'], 3,
                   kwargs['cloud_size']))

    gen[:, :, :gen_cloud.size(2)] = gen_cloud
    sampled_clouds[kwargs['batch_size'] * batch_i:kwargs['batch_size'] * batch_i + gen.shape[0]] = gen

    gt_clouds[kwargs['batch_size'] * batch_i:kwargs['batch_size'] * batch_i + gt_cloud.shape[0]] = gt_cloud

    gt_images[kwargs['batch_size'] * batch_i: kwargs['batch_size'] * batch_i + image.shape[0]] = image
    clouds_file.close()

def train_test(iterator, model, loss_func, optimizer, scheduler, epoch, iter, **config):
    num_workers = config.get('num_workers')
    model_name = config.get('model_name')
    train_mode = config.get('usage_mode')

    batch_time = AverageMeter()
    data_time = AverageMeter()

    torch.set_grad_enabled(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    end = time()
    for i, data in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time() - end)
        scheduler(optimizer, epoch, iter + i)

        input = data[0].to(device)
        seg = data[1].to(device)
        image = data[2].to(device)
        num_seg_classes = data[3][0].to(device)

        segs_batches = []       #every seg classes: has a batch * 3 * npoints_segs_min
        for n in range(1, num_seg_classes + 1):  # for each seg, generate a new batch to train for only a part
            new_batch = []
            min = 2500
            for t in range(len(input)):  # for each batch, create a new seg_batch
                new_seg = []
                k = 0
                for j in range(len(seg[t])):
                    if seg[t][j] == n:
                        new_seg.append(input[t][j])
                        k += 1
                min = k if k <= min and k > 0 else min          # get the least number of points with this seg label

                if k:
                    new_seg = torch.cat(new_seg).reshape(-1, 3).unsqueeze(0)            # get the seg_batch
                    new_seg = torch.transpose(new_seg, 1, 2)
                    new_batch.append(new_seg)

            if new_batch:
                for t in range(len(new_batch)):
                        new_batch[t] = new_batch[t][:, :, :min]  # a new batch with only a seg, with the same size

                new_batch = torch.cat(new_batch, dim=0)  # concat all sge_batches in a batch: batches * 3 * min_seg_num

                if len(new_batch) == len(input):
                    segs_batches.append(new_batch)

        if not segs_batches:
            continue

        image = image.float()  # image: N * 3 * 1377
        full_obj, loss_for_one_epoch = model(segs_batches, image, train_mode, optimizer, loss_func)

        print('[epoch %d][%d / %d]: loss %f' % (epoch, i, len(iterator), loss_for_one_epoch))
        print("full_obj: ", full_obj.size())
        with torch.no_grad():
            if torch.isnan(loss_for_one_epoch):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        batch_time.update(time() - end)

        end = time()

        #if (iter + i + 1) % (100 * num_workers) == 0:
        if (iter + i + 1) % 100 == 0:
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, model_name)

    save_model({
        'epoch': epoch + 1,
        'iter': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, model_name)

def evaluate_test(iterator, model, optimizer, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    saving_mode = kwargs.get('saving_mode')

    model.eval()
    torch.set_grad_enabled(False)

    for i, data in enumerate(iterator):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = data[0].to(device)
        seg = data[1].to(device)
        image = data[2].to(device)
        num_seg_classes = data[3][0].to(device)

        segs_batches = []
        new_batch = torch.zeros((len(input), 3, kwargs['sampled_cloud_size']))
        max = 0
        for t in range(len(input)):
            for j in range(len(seg[t])):
                max = seg[t][j] if seg[t][j] >= max else max        #decide for this batch ,how many NFs should they go through

        if not max:
            continue

        for k in range(max):
            segs_batches.append(new_batch)

        image = image.float()
        with torch.no_grad():
            full_obj, loss = model(segs_batches, image, train_mode, optimizer, loss_func)
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        if saving_mode:
            input = torch.transpose(input, 1, 2)
            save_point_clouds(i, input, full_obj, image, len(iterator), **kwargs)
            print("full_obj: ", full_obj.size())
            print("evaluate: [%d / %d]: loss: %f" % (i, len(iterator), loss))
