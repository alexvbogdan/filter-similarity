# helper function
import sys

import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from dataset import CIFAR10
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def get_network_definition_and_initialize(model_name: str, dataset: str, pretrained: bool = False):
    if model_name == "vgg16":
        from models.vgg import vgg16_bn

        net = vgg16_bn()
    elif model_name == "vgg13":
        from models.vgg import vgg13_bn

        net = vgg13_bn()
    elif model_name == "vgg11":
        from models.vgg import vgg11_bn

        net = vgg11_bn()
    elif model_name == "vgg19":
        from models.vgg import vgg19_bn

        net = vgg19_bn()
    elif model_name == "densenet121":
        from models.densenet import densenet121

        net = densenet121()
    elif model_name == "densenet161":
        from models.densenet import densenet161

        net = densenet161()
    elif model_name == "densenet169":
        from models.densenet import densenet169

        net = densenet169()
    elif model_name == "densenet201":
        from models.densenet import densenet201

        net = densenet201()
    elif model_name == "googlenet":
        from models.googlenet import googlenet

        net = googlenet()
    elif model_name == "inceptionv3":
        from models.inceptionv3 import inceptionv3

        net = inceptionv3()
    elif model_name == "inceptionv4":
        from models.inceptionv4 import inceptionv4

        net = inceptionv4()
    elif model_name == "inceptionresnetv2":
        from models.inceptionv4 import inception_resnet_v2

        net = inception_resnet_v2()
    elif model_name == "xception":
        from models.xception import xception

        net = xception()
    elif model_name == "resnet18":
        from models.resnet import resnet18

        net = resnet18()
    elif model_name == "resnet34":
        from models.resnet import resnet34

        net = resnet34()
    elif model_name == "resnet50":
        from models.resnet import resnet50

        net = resnet50()
    elif model_name == "resnet101":
        from models.resnet import resnet101

        net = resnet101()
    elif model_name == "resnet152":
        from models.resnet import resnet152

        net = resnet152()
    elif model_name == "resnext50":
        from models.resnext import resnext50

        net = resnext50()
    elif model_name == "resnext101":
        from models.resnext import resnext101

        net = resnext101()
    elif model_name == "resnext152":
        from models.resnext import resnext152

        net = resnext152()
    elif model_name == "shufflenet":
        from models.shufflenet import shufflenet

        net = shufflenet()
    elif model_name == "shufflenetv2":
        from models.shufflenetv2 import shufflenetv2

        net = shufflenetv2()
    elif model_name == "squeezenet":
        from models.squeezenet import squeezenet

        net = squeezenet()
    elif model_name == "mobilenet":
        from models.mobilenet import mobilenet

        net = mobilenet()
    elif model_name == "mobilenetv2":
        from models.mobilenetv2 import mobilenetv2

        net = mobilenetv2()
    elif model_name == "nasnet":
        from models.nasnet import nasnet

        net = nasnet()
    elif model_name == "seresnet18":
        from models.senet import seresnet18

        net = seresnet18()
    elif model_name == "seresnet34":
        from models.senet import seresnet34

        net = seresnet34()
    elif model_name == "seresnet50":
        from models.senet import seresnet50

        net = seresnet50()
    elif model_name == "seresnet101":
        from models.senet import seresnet101

        net = seresnet101()
    elif model_name == "seresnet152":
        from models.senet import seresnet152

        net = seresnet152()
    else:
        raise NotImplementedError

    # regarless of the dataset parameter we only want to preload cifar100 weights in our training experiments
    net = network_initialize(net, model_name=model_name, dataset="cifar100", pretrained=pretrained)

    # use the network as feature extractor only
    # this is used only for a single experiment, so not setting a designated script parameter for it
    # set_parameter_requires_grad(net, True)

    # modify last layer according to the dataset
    if dataset == "cifar10":
        net = modify_net_final_layer(net, model_name)

    return net


def network_initialize(net, model_name: str, dataset: str, pretrained: bool = False):
    if pretrained:
        weights_path = f"checkpoint/{dataset}/{model_name}.pth"
        assert os.path.exists(weights_path), f"{weights_path} does not exist"
        net.load_state_dict(torch.load(weights_path))
    return net


def get_network(args):
    """return given network"""
    if args.mode == "transfer":
        net = get_network_definition_and_initialize(model_name=args.net, dataset=args.dataset, pretrained=True)
    elif args.mode == "none":
        net = get_network_definition_and_initialize(model_name=args.net, dataset=args.dataset, pretrained=False)
    else:
        raise NotImplementedError("Mode {args.mode} is not supported yet.")

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(dataset, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """return training dataloader
    Args:
        dataset: str representing dataset name
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if dataset == "cifar100":
        cifar_training = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset == "cifar10":
        cifar_training = CIFAR10(
            root="./data", train=True, download=True, transform=transform_train, num_samples_to_select=1000
        )  # there are 5 subsets, so total number of trianing data = 5 * 1000

    cifar_training_loader = DataLoader(cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_training_loader


def get_test_dataloader(dataset, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """return training dataloader
    Args:
        dataset: str representing dataset name
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == "cifar100":
        cifar_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    elif dataset == "cifar10":
        cifar_test = CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    cifar_test_loader = DataLoader(cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_test_loader


def compute_mean_std(dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
