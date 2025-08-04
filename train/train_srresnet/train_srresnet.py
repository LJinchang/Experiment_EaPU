import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import argparse
import time

import torch
from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.super_resolution.dataset import get_loader, CUDAPrefetcher
from model.srresnet import srresnet_x4
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

from optimizer import AdamNoise, AdamEaPU, SGDNoise, SGDEaPU


# # Random seed to maintain reproducible results
# random.seed(0)
# torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Pytorch SRResNet Training')
parser.add_argument('--train-gt-images-dir', default='./data/SRGAN_ImageNet', type=str, help='train images directory')
parser.add_argument('--gt-image-size', default=96, type=int, help='train images size (cropped)')
parser.add_argument('--test-gt-images-dir', default='./data/Set5/X4/GT', type=str, help='test ground truth directory')
parser.add_argument('--test-lr-images-dir', default='./data/Set5/X4/LR', type=str, help='test low resolution images directory')

parser.add_argument('--batch-size', default=16, type=int, help='train batch size')

parser.add_argument('--optimizer', default='adameapu', help='optimizer type (adameapu, adamnoise, sgdeapu, sgdnoise)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--clip-value', default=1., type=float, help='clip_value = Vclip / Rwg, units: μS')
parser.add_argument('--noise-std', default=0., type=float, help='noise std (the standard deviation of the εcell, units: μS)')
parser.add_argument('--ratio-wg', default=1/80., type=float, help='Rwg')

parser.add_argument('--pretrained-weights-path', default='', help='pretrained model weights path')

parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--train-print-frequency', default=100, type=int, help='train log frequency')
parser.add_argument('--valid-print-frequency', default=1, type=int, help='valid log frequency')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = False
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True

# Model architecture config
model_name = "srresnet_x4"
in_channels = 3
out_channels = 3
channels = 64
num_rcb = 16
# Test upscale factor
upscale_factor = 4
args.upscale_factor = upscale_factor

train_prefetcher, test_prefetcher = get_loader(args, device)
print("Load all datasets successfully.")

model = srresnet_x4(in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels,
                    num_rcb=num_rcb,
                    )
model.to(device)
print(model)
print(f"Build `{model_name}` model successfully.")

criterion = nn.MSELoss()
# loss function weights
loss_weights = 1.0
if args.optimizer.lower() == 'sgdnoise':
    optimizer = SGDNoise(model.parameters(), lr=args.lr, momentum=0.9,
                         noise_std=args.noise_std, ratio_wg=args.ratio_wg)
elif args.optimizer.lower() == 'sgdeapu':
    optimizer = SGDEaPU(model.parameters(), lr=args.lr, momentum=0.9,
                        clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)
elif args.optimizer.lower() == 'adamnoise':
    optimizer = AdamNoise(model.parameters(), lr=args.lr,
                          noise_std=args.noise_std, ratio_wg=args.ratio_wg)
else:  # adameapu
    optimizer = AdamEaPU(model.parameters(), lr=args.lr,
                         clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)

# Initialize the number of training epochs
start_epoch = 0

# Initialize training to generate network evaluation indicators
best_psnr = 0.0
best_ssim = 0.0

print("Check whether to load pretrained model weights...")
if args.pretrained_weights_path:
    model, _, start_epoch, best_psnr, best_ssim, optimizer, _ = load_state_dict(
        model,
        args.pretrained_weights_path,
        optimizer=optimizer,
        load_mode="resume")
    print(f"Loaded `{args.pretrained_weights_path}` pretrained model weights successfully.")
else:
    print("Pretrained model weights not found.")

# Create experiment results
exp_name = f'{model_name}-{args.optimizer}({args.lr}-{args.clip_value}-{args.noise_std}-{args.ratio_wg})'
samples_dir = os.path.join("samples", exp_name)
results_dir = os.path.join("results", exp_name)
make_directory(samples_dir)
make_directory(results_dir)

# Create training process log file
print('Start Tensorboard with "tensorboard --logdir=logs", view at http://localhost:6006/')
writer = SummaryWriter(os.path.join("logs", exp_name))

# Create an IQA evaluation model
psnr_model = PSNR(args.upscale_factor, only_test_y_channel)
ssim_model = SSIM(args.upscale_factor, only_test_y_channel)
psnr_model = psnr_model.to(device)
ssim_model = ssim_model.to(device)


def train_one_epoch(
        srresnet_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        writer: SummaryWriter,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the model in training mode
    srresnet_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        lr = batch_data["lr"].to(device=device, non_blocking=True)

        # Initialize generator gradients
        srresnet_model.zero_grad(set_to_none=True)

        sr = srresnet_model(lr)
        loss = torch.mul(loss_weights, criterion(sr, gt))

        # Backpropagation
        loss.backward()
        # update generator weights
        optimizer.step()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % args.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        srresnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the model in validation mode
    srresnet_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=device, non_blocking=True)
            lr = batch_data["lr"].to(device=device, non_blocking=True)

            # Use the generator model to generate a fake sample
            sr = srresnet_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % args.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == '__main__':
    import numpy as np

    psnrs, ssims = [], []
    start_time = time.strftime('%m-%d-%Y %H:%M:%S')
    psnr, ssim = validate(model,
                          test_prefetcher,
                          -1,
                          writer,
                          psnr_model,
                          ssim_model,
                          'Test'
                          )
    psnrs.append(psnr)
    ssims.append(ssim)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_one_epoch(model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              writer)
        psnr, ssim = validate(model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        psnrs.append(psnr)
        ssims.append(ssim)
        print("\n")

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == args.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict()},
                        f"g_epoch_ckpt.pth",
                        samples_dir,
                        results_dir,
                        "g_best.pth",
                        "g_last.pth",
                        is_best,
                        is_last)
    end_time = time.strftime('%m-%d-%Y %H:%M:%S')
    print('start time:', start_time, '\n', 'end time:', end_time)

    np.save(f'results/record-{args.optimizer}({args.lr}-{args.clip_value}-{args.noise_std}-{args.ratio_wg}).npy',
            {'PSNRs':psnrs, 'SSIMs':ssims})


