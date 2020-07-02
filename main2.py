import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
import pickle
import copy
import torch.utils.model_zoo as model_zoo
from network.critic import *


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def train(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    net = SResNet(is_stigmergy=False, ksai=args.ksai)
    name_net = args.name
    if args.pretrained:
        print("Loading pre-trained model...")
        net.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net = net.cuda(args.cuda_device)
    # 超参数设置
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # 误差函数设置
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer, milestones=[13, 22, 28], gamma=0.1)
    # 数据读入
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    cudnn.benchmark = True

    # 生成数据集
    train_set = datasets.ImageFolder("./data/ILSVRC-12/train", transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_set = datasets.ImageFolder("./data/ILSVRC-12/validate", transform_test)
    validate_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=args.workers)

    # 开始训练
    loss_save = []
    tacc_t1 = AverageMeter()
    tacc_t5 = AverageMeter()
    batch_time = AverageMeter()
    best_t1 = 0.
    best_t5 = 0.
    dic = {}
    dic2 = {}
    net.stigmergy = False
    end = time.time()
    # t1, t5 = validate(validate_loader, net, use_cuda)
    net.train()
    for epoch in range(args.start_epoch, epochs):
        running_loss = 0.0
        lr_scheduler.step()
        if (epoch + 1) == 2:
            net.stigmergy = True
        tacc_t1.reset()
        tacc_t5.reset()
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            outputs = net(b_x, i)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            # 计算loss
            running_loss += loss.item()
            prec1, prec5 = accuracy_imagenet(outputs, b_y, topk=(1, 5))
            tacc_t1.update(prec1[0], size)
            tacc_t5.update(prec5[0], size)
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % 30 == 0:
                print('Train: [{0}-{1}/{2}]\n'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Prec@1 {top1.val:.3f}({top1.avg:.3f})\n'
                      'Prec@5 {top5.val:.3f}({top5.avg:.3f})'.format(
                    epoch+1, i+1, len(train_loader), batch_time=batch_time,
                    top1=tacc_t1, top5=tacc_t5))
        t1, t5 = validate_imagenet(validate_loader, net, use_cuda)
        if t1 > best_t1 or t5 > best_t5:
            best_t1 = t1
            best_t5 = t5
            dic['best_model'] = copy.deepcopy(net.state_dict())
            dic['best_sv'] = copy.deepcopy(net.sv)
            dic['best_dm'] = copy.deepcopy(net.distance_matrices)
        net.train(mode=True)
    with open('./model/record-{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)


if __name__ == "__main__":
    net = "ResNet50-0.3-ImageNet"
    parser = argparse.ArgumentParser()
    parser.add_argument('-pr', type=float, help='pruning rate', default=pr)
    parser.add_argument('--decay', type=float, default=0.6, help='Initialized decay factor in the evaporation process')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
    parser.add_argument('--epochs', type=int, help="training epochs", default=30)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('-cuda_device', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-name', type=str, default='{}'.format(net))
    args = parser.parse_args()
    train(args)





