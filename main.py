# -*-coding:utf-8-*-
from network.architectures import *
from network.critic import *
import argparse
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10 as dataset
import copy
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch
import pickle


def train(args=None):
    assert args is not None
    use_cuda = torch.cuda.is_available() and args.cuda
    # network declaration
    lr_range = []
    if args.architecture == 'ResNet':
        args.epochs = 300
        lr_range = [150, 200, 250, 290]
        net = DcpResNet(num_classes=10, pr=args.pr)
    elif args.architecture == 'Vgg':
        args.epochs = 200
        lr_range = [100, 150, 190]
        net = DcpVgg(pr=args.pr)
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
    # hyper-parameters
    epochs = args.epochs
    lr = args.lr
    batch_size = args.bz
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    lr_scheduler = MultiStepLR(optimizer, milestones=lr_range, gamma=0.1)

    # data load and preprocess
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

    # generate the dataset objects
    train_set = dataset(root=args.data_dir, train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root=args.data_dir, train=False, download=False, transform=transform_test)
    validate_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=args.workers)

    # training begins
    loss_save = []
    tacc_save = []
    vacc_save = []
    best_acc = 0.
    dic = {}
    # update mask
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
            # update masks
            net.update_mask(not net.initialization_over)
            # update the network parameters
            outputs = net(b_x)
            optimizer.zero_grad()
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

            # calculate the loss
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
        print('[%d-%d]\n'
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
    # pruning
    net = pruning(net, args.architecture)
    dic['model'] = net.state_dict() 
    with open('./model/record-{}.p'.format(name_net), 'wb') as f:
        pickle.dump(dic, f)
    



if __name__ == "__main__":
    pr = 0.42
    architecture = 'Vgg'
    parser = argparse.ArgumentParser()
    parser.add_argument('-pr', type=float, help='pruning rate', default=pr)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('-decay', type=float, default=0.6, help='Initialized decay factor in the evaporation process')
    parser.add_argument('--epochs', type=int, help="training epochs", default=200)
    parser.add_argument('--bz', type=int, help='batch size', default=64)
    parser.add_argument('--wd', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--cuda', type=bool, help='GPU', default=True)
    parser.add_argument('-cuda_device', type=int, default=1)
    parser.add_argument('--pre_trained', type=bool, default=False)
    parser.add_argument('--pre_model', type=str, default='record-ResNet32-base-3.p')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--name', type=str, default='ResNet-pr-0.5')
    parser.add_argument('-architecture', type=str, default=architecture)
    parser.add_argument('-data_dir', type=str, default='../data')
    args = parser.parse_args()
    net = "{0}-pr-{1}".format(args.architecture, args.pr)
    args.name = '{}'.format(net)
    train(args)

