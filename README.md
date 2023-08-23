# Robust Mixture-of-Expert Training for Convolutional Neural Networks (ICCV2023)

Official repository for MoE-CNN robust training in our ICCV'23 [paper](https://arxiv.org/abs/2308.10110v1).


## What is in this repository?

This repository supports the robust training of different CNN models using MoE. All the available model architectures are listed in the `models` folder.


## Getting started
Let's start by installing all the dependencies.

```
pip3 install -r requirement.txt
```

### Training

We use `train_moe.py` and `train_ori.py` to adversarially train a original dense model or an MoE model. Argument used in the experiments are stored in `args.py`. The key arguments and their usage are listed below.

* `--arch` This argument specifies the model architecture used for training. There are two categories of models, with their names ending with 'ori' and 'moe' respectively. See all the options in `'models/__init__.py
* `--ratio` The ratio of the MoE and non-MoE model pathways.
* `--n-expert` The number of experts you want to use. This parameter is not valid for `ori` models.
* `--dataset` `CIFAR10 | CIFAR100 | TinyImageNet | ImageNet` Please see below for more detailed dataset preparation for TinyImageNet and ImageNet.
* `--exp-identifier` The sepcial identifier you want to use to differentiate experiment trials. You do not need to use the important paramters (e.g., ratio, n-expert, arch, dataset...) as the identifier, as the folder names automatically contains them.
* `--resume` The path to the checkpoint you want to evaluate or restore training. 
* `--evaluate` Use this parameter to indicate you want to evaluate the checkpoint. Please use `--resume` to indicate the path to the checkpoint you want to evaluate.

### Evaluate with AutoAttack

Please use the file `auto_attack_eval.py` to evaluate the model using AutoAttack. Please use `--source-net` to identify the path to the checkpoint.

### Commands

To train a ResNet-18 MoE model with the expert number of 2 and the ratio of 0.5 on CIFAR-10:
```
python3 train_moe.py --n-expert 2 --arch resnet18_cifar_moe --ratio 0.5 --exp-identifier some_identifier
```

To train a WideResNet-28-10 model with a ratio of 0.5 on CIFAR-100:
```
python3 train_ori.py --dataset CIFAR100 --arch resnet18_cifar_ori --ratio 0.5 --exp-identifier some_identifier
```

To evaluate a VGG-16 MoE model with a ratio of 0.5 on TinyImageNet:

```
python3 train_moe.py --dataset TinyImageNet --evaluate --arch vgg16_bn_moe --ratio 0.5 --n-expert 2 --resume SOME_PATH
```

To evaluate a VGG-16 model with a ratio of 0.5 using AutoAttack on CIFAR-10.

```
python3 auto_attack_eval.py --arch vgg16_bn_ori --ratio 0.5 --source-net SOME_PATH
```


## Dataset Preparation

### ImageNet

The official kaggle website for ImageNet dataset is [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
1. Run  `pip3 install kaggle`
2. Register an account at [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).  
3. Agree the terms and conditions on the [dataset page](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
4. Go to your account page (https://www.kaggle.com//account). Select 'Create API Token' and this will trigger the download of `kaggle.json`, a file containing your API credentials.
5. Copy this file into your server at `~/.kaggle/kaggle.json`.
6. Run command 
   `chmod 600 ~/.kaggle/kaggle.json` and make it visible only to yourself.
7. Run command 
```
kaggle competitions download -c imagenet-object-localization-challenge
```
8. Unzip the file 
```
unzip -q imagenet-object-localization-challenge.zip 
tar -xvf imagenet_object_localization_patched2019.tar.gz
```
9. Enter the validation set folder `cd ILSVRC/Data/CLS-LOC/val`
10. Run script [sh/prepare_imagenet.sh](https://github.com/NormalUhr/Fast_BAT/blob/master/sh/prepare_imagenet.sh) provided by the PyTorch repository, to move the validation subset to the labeled subfolders.

### TinyImageNet

To obtain the original TinyImageNet dataset, please run the following scripts:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -qq 'tiny-imagenet-200.zip'
rm tiny-imagenet-200.zip
```


## Special Credits

Some of the code in this repository is based on the following amazing works.

* https://github.com/allenai/hidden-networks
* https://github.com/inspire-group/hydra





