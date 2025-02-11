import sys

from tqdm import tqdm
import torch


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    right_num = 0
    all_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()

        right_num += torch.eq(torch.argmax(pred, dim=1), labels.to(device)).sum().item()
        all_num += len(images)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] loss:{} acc:{:.3f}".format(epoch, round(mean_loss.item(), 3), right_num / all_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(), right_num / all_num


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    all_num = torch.zeros(1).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    sum_loss = 0

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        all_num += images.size(0)
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        sum_loss += loss.item()
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

        data_loader.desc = '[step:{} loss:{:.3f} acc:{:.3f}]'.format(step, sum_loss / (step + 1), sum_num.item() / all_num.item())

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return sum_loss / (step + 1), sum_num.item() / all_num.item()
