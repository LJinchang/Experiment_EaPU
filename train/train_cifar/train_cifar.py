import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

import torch
from torch import nn

from dataset.cifar import get_loader
from model.preact_resnet import PreActResNet34, PreActResNet152
from optimizer import SGDNoise, SGDEaPU, AdamNoise, AdamEaPU

from collections import defaultdict
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--data-dir', default='./data', type=str, help='path to dataset')
parser.add_argument('--cifar10', default=True, type=bool, help='CIFAR10 or CIFAR100')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--optimizer', default='adameapu', help='optimizer type (adameapu, adamnoise, sgdeapu, sgdnoise)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--clip-value', default=2., type=float, help='clip_value = ΔWth / Rwg, units: μS')
parser.add_argument('--noise-std', default=2., type=float, help='noise std (the standard deviation of the εcell, units: μS)')
parser.add_argument('--ratio-wg', default=1/80., type=float, help='Rwg')

parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Data
trainloader, testloader = get_loader(root=args.data_dir, cifar10=args.cifar10)
num_classes = 10 if args.cifar10 else 100
print("Load all datasets successfully.")

# Model
model_name = 'resnet34'
net = PreActResNet34(num_classes=num_classes)
net = net.to(device)
print(net)
print(f"Build `{model_name}` model successfully.")

criterion = nn.CrossEntropyLoss()
if args.optimizer.lower() == 'sgdnoise':
    optimizer = SGDNoise(net.parameters(), lr=args.lr, momentum=0.9,
                         noise_std=args.noise_std, ratio_wg=args.ratio_wg)
elif args.optimizer.lower() == 'sgdeapu':
    optimizer = SGDEaPU(net.parameters(), lr=args.lr, momentum=0.9,
                        clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)
elif args.optimizer.lower() == 'adamnoise':
    optimizer = AdamNoise(net.parameters(), lr=args.lr,
                          noise_std=args.noise_std, ratio_wg=args.ratio_wg)
else:  # adameapu
    optimizer = AdamEaPU(net.parameters(), lr=args.lr,
                         clip_value=args.clip_value, noise_std=args.noise_std, ratio_wg=args.ratio_wg)
if args.cifar10:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
record = defaultdict(list)

if args.resume:
    # Load checkpoint.
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/ckpt.pt')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('Resuming from checkpoint successfully.')

# Training
def train_one_epoch(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loader = tqdm(trainloader, ncols=100)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if hasattr(optimizer, 'grad_record') and isinstance(optimizer.grad_record, list):
            os.makedirs(f'resnet_grad_lr-{args.lr}', exist_ok=True)
            torch.save(optimizer.grad_record, f'resnet_grad_lr-{args.lr}/grad_epoch-{epoch}.pth')
            optimizer.grad_record = []

        loader.desc = 'Train-[%d|%d] Loss:%.3f Acc:%.3f%% (%d/%d)' % (batch_idx + 1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)

    record['train_acc'].append(round(100.*correct/total, 3))


def validate(epoch, save_checkpoint=True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loader = tqdm(testloader, ncols=100)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loader.desc = 'Test-[%d|%d] Loss:%.3f Acc:%.3f%% (%d/%d)' % (
            batch_idx + 1, len(testloader), test_loss / (batch_idx + 1), 100. * correct / total, correct, total)

        record['val_acc'].append(round(100. * correct / total, 3))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and save_checkpoint:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pt')
        best_acc = acc


if __name__ == '__main__':
    import time
    import numpy as np
    from matplotlib import pyplot as plt

    start_time = time.strftime('%m-%d-%Y %H:%M:%S')
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_one_epoch(epoch)
        validate(epoch, save_checkpoint=True)
        scheduler.step()
    end_time = time.strftime('%m-%d-%Y %H:%M:%S')
    print('start time:', start_time, '\n', 'end time:', end_time)


    plt.rc('font', size=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=16)

    def show(results):
        train_accs = np.array(results['train_acc'])
        test_accs = np.array(results['val_acc'])
        plt.plot(train_accs, label='Train')
        plt.plot(test_accs, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Performance')

        plt.legend()
        plt.tight_layout()
        plt.show()

    show(record)

    os.makedirs('results', exist_ok=True)
    pre_fix = 'cifar10' if args.cifar10 else 'cifar100'
    np.save(f'results/record-{pre_fix}-{args.optimizer}({args.lr}-{args.clip_value}-{args.noise_std}-{args.ratio_wg}).npy', record)

