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
from network.archi_imagenet import *


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
    net = DcpResNet(pr=args.pr)
    name_net = args.name
    if args.pretrained:
        print("Loading pre-trained model...")
        net.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net = net.cuda(args.cuda_device)
    # hyper-parameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # warm-up
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer, milestones=[36, 48, 54], gamma=0.1)
    # data-load 
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

    # generate the data set
    train_set = datasets.ImageFolder(args.data_dir+"/train", transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_set = datasets.ImageFolder(args.data_dir+"/validate", transform_test)
    validate_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=args.workers)

    # training begins
    loss_save = []
    tacc_t1 = AverageMeter()
    tacc_t5 = AverageMeter()
    batch_time = AverageMeter()
    best_t1 = 0.
    best_t5 = 0.
    dic = {}
    net.stigmergy = False
    end = time.time()
    # t1, t5 = validate_imagenet(validate_loader, net, use_cuda)
    net.train()
    net.update_mask(exploration=True)
    for epoch in range(args.start_epoch, epochs):
        tacc_t1.reset()
        tacc_t5.reset()
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # update masks
            net.update_mask(not net.initialization_over)
            outputs = net(b_x)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            # calculate the loss
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
        lr_scheduler.step()
        t1, t5 = validate_imagenet(validate_loader, net, use_cuda)
        if t1 > best_t1 or t5 > best_t5:
            best_t1 = t1
            best_t5 = t5
            dic['channel_utility'] = copy.deepcopy(net.channel_utility)
            dic['architecture'] = copy.deepcopy(net.activated_channels)
            dic['top-1'] = t1
            dic['top-5'] = t5
        net.train(mode=True)
        # the beginning several epochs for warming up the channel-utility
        if (epoch+1) == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 
        if (epoch+1) == 5:
            net.initialization_over = True
    # pruning
    net = pruning(net)
    dic['model'] = net.state_dict()
    with open('./model/record-{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pr', type=float, help='pruning rate', default=0.3)
    parser.add_argument('--decay', type=float, default=0.6, help='Initialized decay factor in the evaporation process')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('--epochs', type=int, help="training epochs", default=60)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('-cuda_device', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--name', type=str, default="ResNet50-0.3-ImageNet")
    parser.add_argument('-data_dir', type=str, default='../data/ILSVRC-12')
    args = parser.parse_args()
    args.name = "ResNet50-{}-ImageNet".format(args.pr)
    train(args)





