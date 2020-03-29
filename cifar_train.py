#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/29 4:28 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import jittor as jt
import os, sys, logging
from jittor import nn
import numpy as np
import jittor.transform as trans
from cifar_dataloader import cifar_dataset
import argparse
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='train cifar10')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--train_size', default=32, type=int, help='train size')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--model',required=True, type=str, help='choose model')
parser.add_argument('--save_path', default='./weights', type=str, help='model save_path')
parser.add_argument('--epochs', default=100, type=int, help='train epochs')
parser.add_argument('--cuda', action='store_true',help='whether to use GPU for training')
parser.add_argument('--visual', action='store_true',help='whether to visual the training state')
args = parser.parse_args()


jt.flags.use_cuda = 1 if args.cuda else 0
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(model, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step(loss)
        if batch_idx % 10 == 0:
            logging.info('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            if writer:
                writer.add_scalar('Train/Loss', loss.data[0], global_step=batch_idx + epoch * len(train_loader))


def test(model, val_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        total_loss += nn.cross_entropy_loss(outputs, targets).data[0]

    logging.info('Total test acc = {}, test loss = {}'.format(total_acc / total_num, total_loss/total_num))
    if writer:
        writer.add_scalar('Test/acc', total_acc / total_num, global_step=epoch)
        writer.add_scalar('Test/loss',total_loss / total_num, global_step=epoch)
    return total_acc / total_num


def choose_model():
    if args.model == 'resnet18':
        from models.resnet18 import ResNet18
        model = ResNet18()
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    return model

def main ():
    logging.info(args)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists('./tensorboard'):
        os.mkdir('./tensorboard')
    writer = SummaryWriter('./tensorboard/' + args.model) if args.visual else None
    batch_size = args.bs
    learning_rate = args.lr
    epochs = args.epochs
    train_transform = trans.Compose([
        trans.RandomCropAndResize(32, scale=(0.5,1)),
        trans.RandomHorizontalFlip(),
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = trans.Compose([
        trans.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = cifar_dataset(is_train=True, transform=train_transform).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = cifar_dataset(is_train=False, transform=test_transform).set_attrs(batch_size=batch_size, shuffle=False)
    model = choose_model()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum= 0.9, weight_decay=1e-4)

    decay_lr_at = [epochs//2,int(epochs//(4/3))]
    for epoch in range(epochs):
        if epoch in decay_lr_at:
            optimizer.lr *= 0.1
        train(model, train_loader, optimizer, epoch, writer)
        if (epoch+1) % 5 == 0:
            acc = test(model, val_loader, (epoch+1), writer)
            model.save(os.path.join(args.save_path, '{}_epoch_{}_acc_{}.pkl'.format(args.model, (epoch+1),acc)))


if __name__ =='__main__':
    main()
