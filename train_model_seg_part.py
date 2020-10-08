import argparse
import os
import io
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.get_segs import ShapeNetDataset

from networks.optimizers import Adam, LRUpdater
from networks.models import Conditional_RNVP_with_image_prior, Model_of_Full_Obj
from networks.losses import Conditional_RNVP_with_image_prior_loss
from training_with_segs import train_test, evaluate_test


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('--config', default='./image_based_model.yaml', help='Path to config file in YAML format.')
    parser.add_argument('--modelname', default='image_prior_rnvp_with_one_part', help='Model name to save checkpoints.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Total number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.000256, help='Learining rate value.')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', default=False, action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')
    parser.add_argument('--predicting', default=False, action='store_true',
                        help='Flag signaling if we do prediction or training')

    parser.add_argument('--class_choice', default='Chair',
                        help='the class of data')
    args = parser.parse_args()
    return args


def save_model(state, model_name):
    torch.save(state, model_name, pickle_protocol=4)
    print('Model saved to ' + model_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

args = define_options_parser()
with io.open(args.config, 'r') as stream:
    config = yaml.load(stream)
config['model_name'] = '{0}.pkl'.format(args.modelname)
config['n_epochs'] = args.n_epochs
config['min_lr'] = config['max_lr'] = args.lr
if args.resume:
    config['resume'] = True
if args.resume_optimizer:
    config['resume_optimizer'] = True
print('Configurations loaded.')

train_dataset = ShapeNetDataset(
    datafile=config['path2data'],
    classification=False,
    class_choice=[args.class_choice])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=config['shuffle'],
    num_workers=int(config['num_workers']))

test_dataset = ShapeNetDataset(
    datafile=config['path2data'],
    classification=False,
    class_choice=[args.class_choice],
    split='test',
    data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=int(config['num_workers']))

print('Dataset init: done.')


num_seg_classes = train_dataset.num_seg_classes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_batch = Conditional_RNVP_with_image_prior(**config).to(device)
model = Model_of_Full_Obj(model_batch, num_seg_classes).to(device)

#if torch.cuda.device_count() > 1:
#    model =  nn.DataParallel(model)        #when do data parallel, errors occur in optimizer.step() so use only one gpu
print('Model init: done.')
print('Total number of parameters: {}'.format(count_parameters(model)))

criterion = Conditional_RNVP_with_image_prior_loss(**config).to(device)


optimizer = Adam(model.parameters(), lr=config['max_lr'], weight_decay=config['wd'],
                 betas=(config['beta1'], config['max_beta2']), amsgrad=True)
scheduler = LRUpdater(len(train_dataloader), **config)
print('Optimizer init: done.')

if not config['resume']:
    cur_epoch = 0
    cur_iter = 0
else:
    path2checkpoint = os.path.join(config['model_name'])
    checkpoint = torch.load(path2checkpoint, map_location='cpu') if device == torch.device('cpu') else torch.load(path2checkpoint)
    cur_epoch = checkpoint['epoch']
    cur_iter = checkpoint['iter']
    print("cur_epoch: ", cur_epoch)
    model.load_state_dict(checkpoint['model_state'])
    if config['resume_optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    del(checkpoint)
    print('Model {} loaded.'.format(path2checkpoint))
    print("resume successfully")
if not args.predicting:
    print("training")
    for epoch in range(cur_epoch, config['n_epochs']):
        train_test(train_dataloader, model, criterion, optimizer, scheduler, epoch, cur_iter, **config)
        cur_iter = 0

else:
    print("predicting")
    with torch.no_grad():
        evaluate_test(test_dataloader, model, optimizer, criterion, **config)