import random
import argparse
import os
import pdb
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from conf import settings
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    AverageMeter,
    WarmUpLR,
    get_network,
    get_test_dataloader,
    get_training_dataloader,
)

torch.multiprocessing.set_sharing_strategy("file_system")


class DiversityLoss(nn.Module):
    def __init__(self, layer_set):
        super(DiversityLoss, self).__init__()
        self.layer_set = layer_set

    def forward(self):
        device = torch.device("cuda" if args.gpu else "cpu")
        layer_losses = []        
        for i, layer in enumerate(self.layer_set):
            if random.random() > args.layer_droprate: # and (layer.weight.shape[2] == 1 or layer.weight.shape[3] == 1):
                similMat = self.getCosineSim(layer)
                layer_sim = torch.norm(similMat)
                if not torch.isnan(layer_sim):
                    layer_losses.append(layer_sim)


        if len(layer_losses) == 0:
            layerwise_diversity = torch.stack([torch.tensor(0, dtype=torch.float32, requires_grad=True, device=device)])
        else:
            layerwise_diversity = torch.stack(layer_losses)
        layerwise_diversity = nn.functional.normalize(layerwise_diversity, dim=0)
        if args.make_similar:
            return  (1 / torch.mean(layerwise_diversity))
        else:
            return  torch.mean(layerwise_diversity)


    def getCosineSim(self, layer, do_abs=False):
        '''
           Compute the cosine-similarity between filters of a conv layer.
           Inputs
           ------
            layer   - 4-D tensor of (numFilters,inputChannels,filterW,filterH).
            do_abs  - Optional boolean flag to take the absval of covar matrix.
           Outputs
           -------
            w_cov   - The pairwise cosine-similarity matrix. 
        '''
        num_filters = layer.weight.shape[0]
        layer_reshaped = layer.weight.view(num_filters, -1)
        layer_normalized = torch.nn.functional.normalize(layer_reshaped,2)
        layer_cov = torch.mm(layer_normalized, torch.t(layer_normalized))

        if do_abs:
            layer_cov = layer_cov.abs()

        return layer_cov


def adjust_alpha(epoch):
    device = torch.device("cuda" if args.gpu else "cpu")

    if args.alpha_grow:
        alpha = torch.tensor(2 / (1 + np.exp(-5 * (float(epoch) / epoch_to_train))) - 1, dtype=torch.float32, device=device)
    elif args.alpha_drop:
        alpha = torch.tensor(-(2 / (1 + np.exp(5 * (float(epoch_to_train - epoch) / epoch_to_train))) - 1), dtype=torch.float32, device=device)
    else:
        alpha = torch.tensor(1, dtype=torch.float32, device=device)

    return alpha


def diversity_loss_fn(output, target, criterion, diversity_loss, alpha):
    loss = criterion(output, target) + diversity_loss * alpha

    ### to check diversity during default training
    # loss = criterion(output, target)

    assert float(loss) >= 0, f"loss cannot be negative: {loss}, {criterion(output, target), {diversity_loss}}"
    return loss, criterion(output, target)


def traverse_net(model, layer_set):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            print(layer, layer.weight.shape)
            if layer in layer_set:
                continue
            layer_set.add(layer)
        else:
            traverse_net(layer, layer_set)


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(diversity_criterion, epoch, reinit, alpha):
    losses = AverageMeter()
    div_losses = AverageMeter()
    top1 = AverageMeter()
    cross = AverageMeter()

    if reinit:
        print("Diversity-mode Training")
    else:
        print("Training")

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_dataloader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        if reinit:
            diversity_loss = diversity_criterion()
            assert not torch.isnan(diversity_loss), "diversity loss is nan"
            loss, cross_loss = diversity_loss_fn(outputs, labels, criterion, diversity_loss, alpha)
            div_losses.update(diversity_loss.item(), images.size(0))
            cross.update(cross_loss.item(), images.size(0))
        else:
            loss = criterion(outputs, labels)

        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(prec1[0], images.size(0))

        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        n_iter = (epoch - 1) * len(training_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        if batch_index % 30 == 0 and batch_index != 0:
            if reinit:
                print(
                    f"Epoch: [{epoch}/{epoch_to_train}][{batch_index}/{len(training_dataloader)}]\t"
                    f"Loss ({losses.avg:.4f})\t"
                    f"Cross ({cross.avg:.4f})\t"
                    f"Div Loss ({div_losses.avg:.4f})\t"
                    f"Alpha ({alpha:.4f})\t"
                    f"Acc@1 {top1.val:.3f}% ({top1.avg:.3f})\t"
                    f"LR {lr:.4f}\t"
                )
            else:
                print(
                    f"Epoch: [{epoch}/{epoch_to_train}][{batch_index}/{len(training_dataloader)}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Acc@1 {top1.val:.3f}% ({top1.avg:.3f})\t"
                    f"LR {lr:.4f}\t"
                )

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

    finish = time.time()

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch):

    print("\nValidation")


    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print("Evaluating Network.....")
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            test_loss / len(test_dataloader.dataset),
            correct.float() / len(test_dataloader.dataset),
            finish - start,
        )
    )
    print()

    # add informations to tensorboard
    writer.add_scalar("Test/Average loss", test_loss / len(test_dataloader.dataset), epoch)
    writer.add_scalar("Test/Accuracy", correct.float() / len(test_dataloader.dataset), epoch)

    return correct.float() / len(test_dataloader.dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument("-dataset", type=str, required=True, help="dataset type: ['cifar10', 'cifar100']")
    parser.add_argument(
        "-mode", type=str, default="none", required=False, help="transfer type: ['transfer', 'none']",
    )
    parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
    parser.add_argument("-b", type=int, default=32, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "-reinit", action="store_true", help="diversity regularization on"
    )
    parser.add_argument(
        "-layer-droprate",
        default=0.33,
        type=float,
        metavar="LAYER_DROP",
        help="portion of layers for comparison (default: 0.33)",
    )
    parser.add_argument(
        "-make-similar", action="store_true", help="push filters to be similar"
    )
    parser.add_argument(
        "-alpha-grow", action="store_true", help="alpha changes from 0 to 1"
    )
    parser.add_argument(
        "-alpha-drop", action="store_true", help="alpha changes from 1 to 0"
    )
    args = parser.parse_args()

    net = get_network(args)
    # data preprocessing:
    if args.dataset == "cifar100":
        training_dataloader = get_training_dataloader(
            args.dataset,
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True,
        )

        test_dataloader = get_test_dataloader(
            args.dataset,
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True,
        )
    else:
        training_dataloader = get_training_dataloader(
            args.dataset,
            settings.TINY_CIFAR10_TRAIN_MEAN,
            settings.TINY_CIFAR10_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True,
        )

        test_dataloader = get_test_dataloader(
            args.dataset,
            settings.TINY_CIFAR10_TRAIN_MEAN,
            settings.TINY_CIFAR10_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            shuffle=True,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.dataset == "cifar100":
        MILESTONES = settings.CIFAR100_MILESTONES
    elif args.dataset == "cifar10" and args.mode == "none":
        MILESTONES = settings.TINY_CIFAR10_MILESTONES
    elif args.dataset == "cifar10" and args.mode == "transfer":
        MILESTONES = settings.TRANSFER_LEARNING_EPOCH_MILESTONES
    else:
        raise NotImplementedError("No config file for this setting yet")

    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=0.2
    )  # learning rate decay

    iter_per_epoch = len(training_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")

    best_acc = 0.0
    if args.mode == "transfer":
        epoch_to_train = settings.TRANSFER_LEARNING_EPOCH
    elif args.dataset == "cifar100":
        epoch_to_train = settings.CIFAR100_EPOCH
    elif args.dataset == "cifar10":
        epoch_to_train = settings.TINY_CIFAR10_EPOCH
    else:
        raise NotImplementedError("No config file for this setting yet")

    print(net)


    layer_set = set()
    if args.reinit:
        traverse_net(net, layer_set)
        assert len(layer_set) != 0, "layer_set must not be empty"

    diversity_criterion = DiversityLoss(layer_set)
    for epoch in range(1, epoch_to_train):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        alpha = adjust_alpha(epoch)
        train(diversity_criterion, epoch, args.reinit, alpha)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01 for original training
        # or start to save best performance model since very beginning in case of transfer learning
        epoch_condition = epoch > 60
        if epoch_condition and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type="best"))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type="regular"))

    writer.close()
