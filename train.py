# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY"S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import sys
import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn 
from datasets.hyspecnet11k import HySpecNet11k
from datasets.SpecEarthData import SpecEarthdata
from datasets.SpecEarthDataLMDB import SpecEarthdataLMDB
from losses import losses
from metrics import metrics
from models import models
from utils import checkpoint
from utils import util
from utils import log
import pickle


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, writer, args):
    model.train()
    device = next(model.parameters()).device

    mse_metric = metrics["mse"]()
    psnr_metric = metrics["psnr"]()
    # ssim_metric = metrics["ssim"]()
    # sa_metric = metrics["sa"]()

    loss_meter = util.AverageMeter()
    mse_meter = util.AverageMeter()
    psnr_meter = util.AverageMeter()
    # ssim_meter = util.AverageMeter()
    # sa_meter = util.AverageMeter()

    org = None
    rec = None

    loop = tqdm(train_dataloader, leave=True)
    loop.set_description(f"Epoch {epoch} Training  ")
    for org in loop:
        org = org.to(device)

        optimizer.zero_grad()
        rec, target, pred_img, mask_img= model(org)
        # print(org.shape)
        if args.loss == 'ghmse':
            out_criterion = criterion(target, rec, mask_img)
        elif args.loss == 'mse_pos_w':
            out_criterion = criterion(target, rec, model)
        else:
            out_criterion = criterion(rec, target)

        if not torch.isnan(out_criterion):
            out_criterion.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

        # compute metrics
        loss = out_criterion
        mse = mse_metric(target, rec)
        psnr = psnr_metric(target, rec)
        # sa = sa_metric(target, rec)
      #  ssim = ssim_metric(target, rec)

        # update metric averages
        loss_meter.update(loss)
        mse_meter.update(mse)
        psnr_meter.update(psnr)
        # sa_meter.update(sa)
     #   ssim_meter.update(ssim)

        # update progress bar to show results of current batch
        loop.set_postfix(
            _loss=loss.item(),
            mse=mse.item(),
            psnr=psnr.item(),
            # sa=sa.item(),
      #      ssim=ssim.item(),
        )

    # get average metrics over whole epoch
    loss_avg = loss_meter.avg.item()
    mse_avg = mse_meter.avg.item()
    psnr_avg = psnr_meter.avg.item()
    # sa_avg = sa_meter.avg.item()
    #ssim_avg = ssim_meter.avg.item()

    # update progress bar to show results of whole training epoch
    loop.set_postfix(
        _loss=loss_avg,
        mse=mse_avg,
        psnr=psnr_avg,
        # sa=sa_avg,
    #    ssim=ssim_avg,
    )

    # log to tensorboard
    if args.loss == 'ghmse':
        log.log_epoch(writer, "train", epoch, loss_avg, mse_avg, psnr_avg, org, rec, target, rec, pred_img, model)
    else:
        log.log_epoch(writer, "train", epoch, loss_avg, mse_avg, psnr_avg, org, rec, target, pred_img, mask_img, model)


def val_epoch(epoch, val_dataloader, model, criterion, writer,args):
    model.eval()
    device = next(model.parameters()).device

#    bpppc = model.bpppc

    mse_metric = metrics["mse"]()
    psnr_metric = metrics["psnr"]()
 #   ssim_metric = metrics["ssim"]()
    # sa_metric = metrics["sa"]()

    loss_meter = util.AverageMeter()
    mse_meter = util.AverageMeter()
    psnr_meter = util.AverageMeter()
    # sa_meter = util.AverageMeter()
    # ssim_meter = util.AverageMeter()

    with torch.no_grad():
        loop = tqdm(val_dataloader, leave=True)
        loop.set_description(f"Epoch {epoch} Validation")
        for org in loop:
            org = org.to(device)

            rec, target, pred_img, mask_img= model(org)
            # print(rec.shape)
            if args.loss == 'ghmse':
                out_criterion = criterion(target, rec, mask_img)
            elif args.loss == 'mse_pos_w':
                out_criterion = criterion(target, rec, model)
            else:
                out_criterion = criterion(rec, target)

            # compute metrics
            loss = out_criterion
            mse = mse_metric(target, rec)
            psnr = psnr_metric(target, rec)
            # sa = sa_metric(target, rec)
            # ssim = ssim_metric(target, rec)

            # update metric averages
            loss_meter.update(loss)
            mse_meter.update(mse)
            psnr_meter.update(psnr)
            # sa_meter.update(sa)
            # ssim_meter.update(ssim)

            # update progress bar to show results of current batch
            loop.set_postfix(
                _loss=loss.item(),
                # bpppc=bpppc,
                mse=mse.item(),
                psnr=psnr.item(),
                # sa=sa.item(),
                # ssim=ssim.item(),
            )

        # get average metrics over whole validation set
        loss_avg = loss_meter.avg.item()
        mse_avg = mse_meter.avg.item()
        psnr_avg = psnr_meter.avg.item()
        # sa_avg = sa_meter.avg.item()
        # ssim_avg = ssim_meter.avg.item()

        # update progress bar to show results of whole validation set
        loop.set_postfix(
            _loss=loss_avg,
            # bpppc=bpppc,
            mse=mse_avg,
            psnr=psnr_avg,
            # sa=sa_avg,
            # ssim=ssim_avg,
        )

        # log to tensorboard
        # log.log_epoch(writer, "val", epoch, loss_avg, bpppc, mse_avg, psnr_avg, ssim_avg, sa_avg, org, rec)
        if args.loss == 'ghmse':
            log.log_epoch(writer, "val", epoch, loss_avg, mse_avg, psnr_avg, org, rec, target, rec, pred_img,model)
        else:
            log.log_epoch(writer, "val", epoch, loss_avg, mse_avg, psnr_avg, org, rec, target, pred_img, mask_img,model)

    return loss_avg

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    save_dir = util.get_save_dir(args)
    writer = SummaryWriter(log_dir=save_dir)
    if args.dataset=='hyspecnet':
        train_dataset =    HySpecNet11k("./datasets/hyspecnet-11k/", mode=args.mode, split="train", transform=None)
        val_dataset = HySpecNet11k("./datasets/hyspecnet-11k/", mode=args.mode, split="val", transform=None)
    elif args.dataset=='SpectralEarth':
        train_dataset =    SpecEarthdata("./datasets/spectral_earth", split="train", transform=None)
        val_dataset = SpecEarthdata("./datasets/spectral_earth", split="val", transform=None)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    device = f"cuda:{args.devices[0][0]}" if args.devices[0] != "cpu" and torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    

    net = models[args.model](src_channels=args.num_channels, mask_ratio = args.masking_ratio, patch_size=args.patch_size,
            vit_model=args.vit_model,)
    net = net.to(device)

    if args.devices != "cpu" and len(args.devices[0]) > 1 and torch.cuda.device_count() > 1:
        net = util.CustomDataParallel(net, device_ids=list(map(int, args.devices[0].split(','))))
    #net = util.CustomDataParallel(net, device_ids=[2,3,4])
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    
    criterion = losses[args.loss]()

    last_epoch = 0
    # load from previous checkpoint
    if args.checkpoint:
        last_epoch = checkpoint.load_checkpoint_train(args.checkpoint, net, optimizer, device)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            args
        )

        loss = val_epoch(epoch, val_dataloader, net, criterion, writer,args)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
        }
        checkpoint.save_checkpoint(state, is_best, save_dir=save_dir)

    # save weights of the best epoch without additional information
    checkpoint.strip_checkpoint(f"{save_dir}best.pth.tar", save_dir=save_dir)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for the Vision Transformer (default: %(default)s)"
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="vit_base_patch16_224",
        help="ViT model type (e.g., vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224)"
    )
    parser.add_argument(
        "--dic-channels",
        type=int,
        default=100,
        help="Number of dictionary channels for the decoder (default: %(default)s)"
    )
    parser.add_argument(
        "--weight-file",
        type=str,
        default=None,
        help="Path to the weight file for decoder2 initialization (default: %(default)s)"
    )
    parser.add_argument(
        "--fixed-weights",
        action="store_true",
        help="If set, decoder2 weights will remain fixed during training"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=[0],
        nargs="+",
        help="Devices to use (default: %(default)s), e.g. cpu or 0 or 0,2,5,7 for multiple"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
        help="Training batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=4,
        help="Validation batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Data loaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./datasets/hyspecnet-11k/",
        #default="/media/storagecube/data/shared/datasets/enmap/dataset/",
        help="Path to dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="Dataset split difficulty (default: %(default)s)"
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=202,
        help="Number of data channels, (default: %(default)s)"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mae",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=losses.keys(),
        type=str,
        help="Loss (default: %(default)s)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-mr",
        "--masking-ratio",
        default=0.75,
        type=float,
        help="Masking rate (default: %(default)s)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results/trains/",
        help="Directory to save results (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10587,
        help="Set random seed for reproducibility (default: %(default)s)"
    )
    parser.add_argument(
        "--clip-max-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training"
    )
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=0,
        help="Sparsity regularization parameter (default: %(default)s)"
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=0.5,
        help="Contrastive loss weight (default: %(default)s)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss (default: %(default)s)"
    )
    parser.add_argument(
        "--mv_lambda",
        type=float,
        default=0,
        help="Sparsity regularization parameter (default: %(default)s)"
    )
    parser.add_argument(
        "--group",
        type=int,
        default=None,
        help="band grouping (default: %(default)s)"
    )
    parser.add_argument('--adaptive_masking', action='store_true',
                   help='Enable adaptive masking')
    parser.add_argument(
        "--use-spectral-reduction",
        action='store_true',
        help="Use spectral reduction in SpecMAE"
    )
    parser.add_argument(
        "--attention-strength",
        type=float,
        default=0.1,
        help="Strength of the attention mechanism in MAE (default: %(default)s)"
    )
    parser.add_argument(
        "--constraint-type",
        type=str,
        default="none",
        choices=["abs", "square", "none", "sigmoid"],
        help="Type of constraint for the decoder (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(sys.argv[1:])
