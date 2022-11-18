import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import models_self
from meter import *

from torch.utils.tensorboard import SummaryWriter
writer1 = SummaryWriter('/output/logs/trainloss')
writer2 = SummaryWriter('/output/logs/trainacc1')
writer3 = SummaryWriter('/output/logs/trainacc5')


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    train_loss = 0.0
    train_acc1 = 0.0
    train_acc5 = 0.0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #tensorboard show loss/acc
        train_loss += loss.item()
        train_acc1 += acc1[0]
        train_acc5 += acc5[0]
        if i%100 == 99:
            writer1.add_scalar('train loss',train_loss/100,epoch*len(train_loader)+i)
            train_loss = 0.0

            writer2.add_scalar('train acc1',train_acc1/100,epoch*len(train_loader)+i)
            train_acc1 = 0.0

            writer3.add_scalar('train acc5',train_acc5/100,epoch*len(train_loader)+i)
            train_acc5 = 0.0
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)