import argparse
import os
import sys
import random
import json
import numpy as np
import torch

from typing import Iterable, Optional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
import util
from models import build_model 
from datasets import build_dataset
from loss import build_criterion, criterion 
from common.error import NoGradientError
from common.logger import Logger, MetricLogger, SmoothedValue
from common.functions import *
from common.nest import NestedTensor
from configs import dynamic_load
import cv2

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(
    loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module, print_freq=10., tb_logger=None
):
    model.eval()
    def _transform_inv(img,mean,std):
        img = img * std + mean
        img  = np.uint8(img * 255.0)
        img = img.transpose(1,2,0)
        return img


    logger = MetricLogger(delimiter=' ')
    header = 'Test'

    for sample_batch in logger.log_every(loader, print_freq, header):
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix = sample_batch['gt_matrix'].cuda().float()

        preds = model(images0, images1, gt_matrix)
        targets = {'gt_matrix': gt_matrix}
        loss_dict = criterion(preds, targets)
        loss = loss_dict['losses']
        print(loss)

def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', n_params)
    model = model.to(DEV)
    train_dataset, test_dataset = build_dataset(args)
    test_sampler = SequentialSampler(test_dataset)
    model_without_ddp = model


    dataloader_kwargs = {
        #'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 4,
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    state_dict = torch.load("artifacts/resnet101-dual_softmax_dim256-128_depth256-128/model_th0.1_best_3428.123.pth", map_location='cpu')
    model_without_ddp.load_state_dict(state_dict['model'])

    criterion = build_criterion(args)
    criterion = criterion.to(DEV)

    print('Start Testing...')

    test_stats = test(
        test_loader,
        model,
        criterion,
    )
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('=='*40 + '\n')

    main(args)
