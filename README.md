# Dynamic Channel Propagation
The code for the contributed paper "Learning to Prune in Training via Dynamic Channel Propagation" accepted by [ICPR-2020](https://www.micc.unifi.it/icpr2020/). In this paper, we propose a novel network training mechanism called "dynamic channel propagation" to prune the deep neural networks during the training period. Here we show the source code of our scheme.
## Preliminalies
### Environment
Our code is based on the deep-learning framework [Pytorch](https://pytorch.org/) and strongly reference to its official [examples](https://github.com/pytorch/examples). 
* python >= 3.5
* cuda >= 10.0
* torch >= 1.3.0, torchvision
### Data set
For CIFAR-10, you may directly download it using pytorch API
```python
from torchvision.datasets.cifar import CIFAR10 as dataset
# for training set
dataset(root='../data', train=True, download=True)
# for testing set
dataset(root='../data', train=False, download=True)
```
As for ILSVRC-2012(ImageNet), you have to download it from the [URL](http://image-net.org/challenges/LSVRC/2012/index), unzip it and move the validating images to subfolders by the [shell](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Run the code
Firstly, enter the root directory of the project, and then generate a folder to store the results
```
cd [root directory of the project]
mkdir model
```
Type the following code to run on CIFAR-10
```
pyhton3 main.py -architecture [Vgg or ResNet] -decay [initial value of decay factor] -pr [global pruning rate of channels]
```
In regard with pruning ResNet on ILSVRC-2012, type
```
pyhton3 main2.py -pr [global pruning rate of channels]
```

