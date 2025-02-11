import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math
import argparse
import time

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model.swin_transformer import swin_tiny_patch4_window7_224, SwinTransformer
from optimizer import SGDNoise, SGDEaPU, AdamNoise, AdamEaPU
import torch.optim.lr_scheduler as lr_scheduler

from dataset.mini_imagenet import MiniImageNet
from train_eval_utils import train_one_epoch, validate

from collections import defaultdict

from torch.backends import cudnn
cudnn.benchmark = False


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=logs", view at http://localhost:6006/')
    post_fix = '-transfer' if args.freeze_layers else ''
    exp_name = f'swin_transformer-{args.optimizer}({args.lr}-{args.clip_value}-{args.noise_std}-{args.ratio_wg}){post_fix}'
    tb_writer = SummaryWriter(os.path.join("logs", exp_name))
    if os.path.exists("./checkpoint") is False:
        os.makedirs("./checkpoint")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    json_path = "./mini_imagenet/classes_name.json"

    train_data_set = MiniImageNet(root_dir=data_root,
                               csv_name="new_train.csv",
                               json_path=json_path,
                               transform=data_transform["train"])

    # check num_classes
    if args.num_classes != len(train_data_set.labels):
        raise ValueError("dataset have {} classes, but input {}".format(len(train_data_set.labels),
                                                                        args.num_classes))

    val_data_set = MiniImageNet(root_dir=data_root,
                             csv_name="new_val.csv",
                             json_path=json_path,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    print("Load all datasets successfully.")

    # create model
    model = swin_tiny_patch4_window7_224(num_classes=args.num_classes).to(device)
    print(model)
    print(f"Build swin transformer model successfully.")

    model.to(device)
    n = 0
    for p in model.parameters():
        n += p.numel()
    print('params:', n)

    if isinstance(model, SwinTransformer):
        if args.pretrained_weights_path != "":
            assert os.path.exists(args.pretrained_weights_path), "weights file: '{}' not exist.".format(args.pretrained_weights_path)
            weights_dict = torch.load(args.pretrained_weights_path, map_location=device)["model"]
            # delete classifier head
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
            print(f"Loaded `{args.pretrained_weights_path}` pretrained model weights successfully.")

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # freeze all the parameters expect head
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        # args.lr = 1e-5
        # args.lrf = 1e-6
        # args.epochs = 20
    else:
        if os.path.exists(args.pretrained_weights_path):
            weights_dict = torch.load(args.pretrained_weights_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            missing_keys, unexpected_keys = model.load_state_dict(load_weights_dict, strict=False)
            print('missing_keys:', missing_keys)
            print('unexpected_keys:', unexpected_keys)

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "fc" not in name:
                    para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    global optimizer

    if args.optimizer.lower() == 'sgdnoise':
        optimizer = SGDNoise(pg, lr=args.lr, momentum=0.9,
                             noise_std=args.noise_std, ratio_wg=args.ratio_wg)
    elif args.optimizer.lower() == 'sgdeapu':
        optimizer = SGDEaPU(pg, lr=args.lr, momentum=0.9,
                            clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)
    elif args.optimizer.lower() == 'adamnoise':
        optimizer = AdamNoise(pg, lr=args.lr,
                              noise_std=args.noise_std, ratio_wg=args.ratio_wg)
    else:  # adameapu
        optimizer = AdamEaPU(pg, lr=args.lr,
                             clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)

    if isinstance(model, SwinTransformer):
        scheduler = lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)
    else:
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        validate(model, val_loader, device)
        return

    best_acc = 0
    record = defaultdict(list)
    # initialization metrics
    train_loss, train_acc = validate(model=model,
                                     data_loader=train_loader,
                                     device=device
                                     )
    val_loss, val_acc = validate(model=model,
                                 data_loader=val_loader,
                                 device=device
                                 )
    record['train_acc'].append(train_acc)
    record['val_acc'].append(val_acc)
    tb_writer.add_scalars('loss',
                          {'train_loss': train_loss,
                           'val_loss': val_loss
                           },
                          0)
    tb_writer.add_scalars('accuracy',
                          {'train_acc': train_acc,
                           'val_acc': val_acc
                           },
                          0)

    tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], 0)
    start_time = time.strftime('%m-%d-%Y %H:%M:%S')
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = validate(model=model,
                                     data_loader=val_loader,
                                     device=device)

        record['train_acc'].append(train_acc)
        record['val_acc'].append(val_acc)
        tb_writer.add_scalars('loss',
                              {'train_loss': train_loss,
                               'val_loss': val_loss
                               },
                              epoch + 1)
        tb_writer.add_scalars('accuracy',
                              {'train_acc': train_acc,
                               'val_acc': val_acc
                               },
                              epoch + 1)

        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch + 1)

        if best_acc < val_acc:
            print('Saving...')
            best_acc = val_acc
            torch.save(model.state_dict(), "./checkpoint/swin_transformer_tiny.pth")
    end_time = time.strftime('%m-%d-%Y %H:%M:%S')
    print('start time:', start_time, '\n', 'end time:', end_time)

    os.makedirs('results', exist_ok=True)
    np.save(f'results/record-{args.optimizer}({args.lr}-{args.clip_value}-{args.noise_std}-{args.ratio_wg}).npy',
            record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--optimizer', default='adameapu',
                        help='optimizer type (adameapu, adamnoise, sgdeapu, sgdnoise)')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lrf', type=float, default=1e-6)
    parser.add_argument('--clip-value', default=2., type=float, help='clip_value = Vclip / Rwg, units: μS')
    parser.add_argument('--noise-std', default=0., type=float,
                        help='noise std (the standard deviation of the εcell, units: μS)')
    parser.add_argument('--ratio-wg', default=1 / 80., type=float, help='Rwg')

    # data_root
    parser.add_argument('--data-path', type=str, default="./mini_imagenet")

    parser.add_argument('--pretrained_weights_path', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    default = 'checkpoint/swin_transformer_tiny.pth'
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--freeze-layers', type=bool, default=False)

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    optimizer = None
    main(opt)
