# -*-coding:utf-8-*-
from network.architectures import *
import argparse
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10 as dataset
import copy
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch


def train(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    lr_range = []

    if args.architecture == 'ResNet':
        args.epochs = 300
        lr_range = [150, 200, 250, 290]
        net = SResNet(num_classes=10, pr=args.pr)
    elif args.architecture == 'Vgg':
        args.epochs = 200
        lr_range = [100, 150, 190]
        net = SVgg(pr=args.pr)
        # net = CompactVgg(pr=args.pr)
    else:
        exit(0)
    if args.pre_trained is True:
        args.lr = 0.01
        lr_range = [100, 150, 190]
        net.decay = 0.7
        args.epochs = 200
    name_net = args.name
    print(args.decay)

    if use_cuda:
        torch.cuda.set_device(args.cuda_device)
        net.cuda(device=args.cuda_device)
    if args.pre_trained:
        print("Load post-training models...")
        with open('./pre-trained/{}'.format(args.pre_model), 'rb') as f:
            d = pickle.load(f)
            net.load_state_dict(d['best_model'])
    # 超参数设置
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # 误差函数设置
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    lr_scheduler = MultiStepLR(optimizer, milestones=lr_range, gamma=0.1)

    # 数据读入
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 生成数据集
    train_set = dataset(root='../data', train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root='../data', train=False, download=False, transform=transform_test)
    validate_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=args.workers)

    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    best_acc = 0.
    dic = {}
    # 第一次更新掩码
    net.update_mask(exploration=True)
    for epoch in range(epochs):
        running_loss = 0.0
        correct_count = 0.
        count = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            net.update_mask(not net.initialization_over)
            # 网络更新部分
            outputs = net(b_x)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 50 == 0:
                print(
                    '[ %d-%d ] loss: %.9f, \n'
                    'training accuracy: %.6f' % (
                    epoch+1, i + 1, running_loss / count,
                    correct_count / count))
                loss_save.append(running_loss / count)
                tacc_save.append(correct_count / count)
        lr_scheduler.step()
        net.train(mode=False)
        acc = validate(net, validate_loader, use_cuda, device=args.cuda_device)
        print('[ %d-%d]\n'
              'validating accuracy: %.6f' % (epoch+1, epochs, acc))
        vacc_save.append(acc)
        if acc > best_acc:
            print("Better accuracy : {}".format(acc))
            best_acc = acc
            dic['best_model'] = copy.deepcopy(net.state_dict())
            dic['best_cu'] = copy.deepcopy(net.channel_utility)
        net.train(mode=True)
        # the beginning several epochs for warm-up of channel-utility
        if (epoch+1) == 10:
            net.initialization_over = True

    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    dic['channel_utility'] = net.channel_utility
    dic['architecture'] = net.activated_channels
    with open('./model/record-{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)


def test(args=None):
    assert args is not None
    torch.cuda.set_device(args.cuda_device)
    use_cuda = True
    # net = Vgg()
    # net = Svgg()
    net = SResNet()
    with open('./model/record-ResNet56-0.3-3-decay-0.8.p', 'rb') as f:
        dic = pickle.load(f)
        # net.load_state_dict(dic['best_model'])
        # net.channel_utility = dic['best_cu']
        # net.relevance_matrices = dic['best_rm']
        print(max(dic['validating_accuracy']))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # val_set = dataset(root='./data', train=False, download=False, transform=transform_test)
    # loader = DataLoader(val_set, batch_size=256, shuffle=True, num_workers=args.workers)
    # net.eval()
    # acc = validate(net, loader, use_cuda=use_cuda, device=args.cuda_device)
    # print("testing accuracy : {}".format(acc))
    return


if __name__ == "__main__":
    prs = [0.1, 0.2, 0.3, 0.5, 0.6]
    pr = 0.42
    architecture = 'Vgg'
    pre_trained = False
    net = "".format(pr)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr', type=float, help='pruning rate', default=pr)
    parser.add_argument('-mode', type=str, help='training or testing')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('-decay', type=float, default=0.6, help='Initialized decay factor in the evaporation process')
    parser.add_argument('--epochs', type=int, help="training epochs", default=200)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('-cuda_device', type=int, default=1)
    parser.add_argument('--pre_trained', type=bool, default=pre_trained)
    parser.add_argument('--pre_model', type=str, default='record-ResNet32-base-3.p')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--name', type=str, default='{}'.format(net))
    parser.add_argument('--architecture', type=str, default=architecture)
    args = parser.parse_args()
    for pr in [0.4, 0.5, 0.6]:
        args.pr = pr
        for epoch in [1, 2, 3]:
            net = "Vgg16-{0}-iteration-{1}".format(pr, epoch)
            args.name = '{}'.format(net)
            print("{}.".format(args.name))
            train(args)
    # for epoch in [1, 2, 3]:
    #     net = "ResNet32-{0}-{1}".format(pr, epoch)
    #     args.name = '{}'.format(net)
    #     print("{}.".format(args.name))
    #     train(args)
