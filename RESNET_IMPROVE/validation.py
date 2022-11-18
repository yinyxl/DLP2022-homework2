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
writer1 = SummaryWriter('/output/logs/testloss')
writer2 = SummaryWriter('/output/logs/testacc1')
writer3 = SummaryWriter('/output/logs/testacc5')

def validate(val_loader, model, criterion, args,epoch):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            test_loss = 0.0
            test_acc1 = 0.0
            test_acc5 = 0.0

            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available():
                #     images = images.to('mps')
                #     target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # tensorboard show loss/acc
                test_loss += loss.item()
                test_acc1 += acc1[0]
                test_acc5 += acc5[0]
                if i%args.print_freq == 0:
                    writer1.add_scalar('test loss',test_loss/args.print_freq,epoch*len(loader)+i)
                    test_loss = 0.0

                    writer2.add_scalar('test acc1',test_acc1/args.print_freq,epoch*len(loader)+i)
                    test_acc1 = 0.0

                    writer3.add_scalar('test acc5',test_acc5/args.print_freq,epoch*len(loader)+i)
                    test_acc5 = 0.0

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg