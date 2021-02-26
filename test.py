import argparse

import torch
from conf import settings
from torch.utils.data import DataLoader
from utils import get_network, get_test_dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-mode",
        type=str,
        default="none",
        required=False,
        help="it is here just to follow common interface for get_network(args) "
        "This will ensure that network will be loaded without pretrained weights, "
        "and we load weights directly through -weights parameters.",
    )
    parser.add_argument("-dataset", type=str, required=True, help="dataset type: ['cifar10', 'cifar100']")
    parser.add_argument("-gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument("-b", type=int, default=16, help="batch size for dataloader")
    args = parser.parse_args()

    assert args.mode == "none", "test mode does not support any transfer methods in this work"

    net = get_network(args)

    if args.dataset == "cifar100":
        test_loader = get_test_dataloader(
            args.dataset,
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            # settings.CIFAR100_PATH,
            num_workers=4,
            batch_size=args.b,
        )
    else:
        test_loader = get_test_dataloader(
            args.dataset,
            settings.TINY_CIFAR10_TRAIN_MEAN,
            settings.TINY_CIFAR10_TRAIN_STD,
            # settings.CIFAR100_PATH,
            num_workers=4,
            batch_size=args.b,
        )

    net.load_state_dict(torch.load(f"checkpoint/{args.net}.pth"))
    # print(net)
    print()
    print(args.net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    print("Top 1 acc: ", correct_1 / len(test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
