# configurations for this project

import os
from datetime import datetime

# CIFAR100 dataset path (python version)
# CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

# mean and std of tiny_cifar10 dataset
TINY_CIFAR10_TRAIN_MEAN = (0.4260, 0.4164, 0.3850)
TINY_CIFAR10_TRAIN_STD = (0.2543, 0.2494, 0.2423)

# mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_TRAIN_STD = (0.2673, 0.2564, 0.2762)

# directory to save weights file
CHECKPOINT_PATH = "checkpoint"

# CIFAR100 total training epoches
CIFAR100_EPOCH = 200
CIFAR100_MILESTONES = [60, 120, 160]

# TINY_CIFAR10 total training epoches
TINY_CIFAR10_EPOCH = 100
TINY_CIFAR10_MILESTONES = [30, 60, 80]

# transfer learning milestones
TRANSFER_LEARNING_EPOCH = 20
TRANSFER_LEARNING_EPOCH_MILESTONES = [10, 15, 18]

# time of we run the script
TIME_NOW = datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")

# tensorboard log dir
LOG_DIR = "runs"

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
