## classification.jittor
Classification on CIFAR-10/100 and ImageNet with [Jittor](https://github.com/Jittor/jittor).

## Features
* Unified interface for different network architectures
* Training progress bar with rich info
* visualize the training state.

## Install 
1. Download the CIFAT-10 Dataset and put it into `./data/cifar`
2. Run commond to run
```shell
python cifar_train.py --model resnet18 --cuda
```
## Result

Top1 error rate on the CIFAR-10/100 benchmarks are reported. .
* Note that the number of parameters are computed on the **CIFAR-10 dataset**.

 | Model  | CIFAR-10 (%) | CIFAR-100 (%)|
 |:---:|:---:|:---:|
 |ResNet18|92.04%|-|
 |SENet18|91.58%|-|
