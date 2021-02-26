# filter-similarity

## CIFAR10 model weights
Use [this](https://rutgers.box.com/shared/static/y9wi8ic7bshe2nn63prj9vsea7wibd4x.zip) link to download weights (you have to download and extract manually).
To set the path to the downloaded weights folder specify `path_to_state_dict` in `vgg.py`.

## Dependencies
Install dependecies by running:
```bash
pip install -r requirements.txt
```

### Options:
```-net``` - choice of architecture (default: resnet18)  
```-dataset``` - choice of dataset (default: cifar10)  
```-tr-batch``` - training batch size (default: 512)  
```-val-batch``` - validation batch size (default: 512)  
```-lr``` - learning rate (default: 0.1)  
```-wd``` - weight decay (default: 5e-4)  
```-epochs``` - number of epochs to train (default: 300)  
```-cpu``` - cpu flag  
```-reinit``` - diversity loss usage flag  
```-mode``` - choice between transfer and default training  

### How to run:
```bash
python train.py -lr 0.1 -gpu -dataset cifar100 -reinit -alpha-drop -net resnet18 -b 512

python train.py -lr 0.1 -gpu -dataset cifar100 -net resnet18 -b 512

```

## Results


| No. |     Model    |  default  |     cosine    |  dataset  |    alpha    |  epochs  |
|:---:|:------------:|----------:|:-------------:|:---------:|:-----------:|:--------:|
| 1   | resnet18     |   75.21%  |     75.82%    | cifar100  | from 1 to 0 |    200   |
| 2   | resnet34     |   75.54%  |     76.91%    | cifar100  | from 1 to 0 |    200   |
| 3   | resnet50     |   75.52%  |     78.14%    | cifar100  | from 1 to 0 |    200   |
| 4   | seresnet18   |   74.56%  |     75.58%    | cifar100  | from 1 to 0 |    200   |
| 5   | mobilenet    |   67.03%  |     67.25%    | cifar100  | from 1 to 0 |    200   |
| 6   | mobilenetv2  |   69.53%  |     69.72%    | cifar100  | from 1 to 0 |    200   |
| 7   | squeezenet   |   68.91%  |     69.44%    | cifar100  | from 1 to 0 |    200   |




## How to contribute
We use pre-commit hooks with black and flake8 to align code format.
```bash
pip install pre-commit black flake8
```

Initialize it from the folder with the repo:
```bash
pre-commit install
```
