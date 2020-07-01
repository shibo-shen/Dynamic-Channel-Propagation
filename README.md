# Dynamic Channel Propagation
The code for the contributed paper "Learning to Prune in Training via Dynamic Channel Propagation" accepted by ICPR-2020
## Preliminalies
### Environment
The code is based on the deep-learning framework [Pytorch](https://pytorch.org/) and strongly reference to official [code](https://github.com/pytorch/pytorch). 
* cuda >= 10.0
* pytorch >= 1.3.0
### Data set
For CIFAR-10 data set, you may directly download it using the pytorch API
'''python
# for training set
dataset(root='../data', train=True, download=True, transform=transform_train)
# for testing set
dataset(root='../data', train=False, download=True, transform=transform_train)
'''

## Run the code

## Result
