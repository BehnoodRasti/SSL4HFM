import os
import torch

from torch import nn

from utils import time


def get_save_dir(args):
    if args.checkpoint is None:
        save_dir = f"{args.save_dir}{time.get_date_time()}-{get_run_name(args)}/"
    else:
        save_dir = f"{os.path.dirname(args.checkpoint)}/"
    return save_dir


def get_run_name(args):
    save_dir = f"{args.model}-{args.loss}-{args.mode}-{args.learning_rate}-{args.l1_lambda}-{args.l2_lambda}-{args.mv_lambda}-{args.temperature}-{args.masking_ratio}"
    return save_dir


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not torch.isnan(val):
            val = val.data
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
