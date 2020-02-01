
We apply the recent [Filter Response Normalization](https://arxiv.org/abs/1911.09737)
method on a better and common training recipe of ResNet-50 on ImageNet,
to understand how well it works under this recipe.

## Baseline Code and Training Recipe

We take the ImageNet training code in TensorFlow from
[ppwwyyxx/GroupNorm-reproduce](https://github.com/ppwwyyxx/GroupNorm-reproduce/tree/master/ImageNet-ResNet-TensorFlow).
The training code follows a common training recipe that is used in the following two papers:
* [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
* [Group Normalization](https://arxiv.org/abs/1803.08494)

and reproduces exact baselines numbers of the above two papers, i.e.:

| Model                | Top 1 Error |
|:---------------------|:------------|
| ResNet-50, BatchNorm | 23.6%       |
| ResNet-50, GroupNorm | 24.0%       |


We apply a patch [FRN-cosineLR.diff](FRN-cosineLR.diff) on the abovementioned code on top of commit
[e0d0b1](https://github.com/ppwwyyxx/GroupNorm-reproduce/commit/e0d0b152d3061f25d80df29080cb3cceedad3f5a), to implement
Filter Response Normalization as well as cosine LR schedule.

The updated code is included in this directory.

## Run:

This command trains a ResNet-50 with BatchNorm on ImageNet:
```
./imagenet-resnet.py --data /path/to/imagenet
```

To use FRN+TLU, add `--frn-trelu`. To use cosine LR schedule, add `--cosine-lr`.

## Results:
We train our models on machines with 8 V100s using TensorFlow 1.14.

Without cosine LR schedule:

| Model               | Top 1 Error |
|:--------------------|:------------|
| ResNet-50, BN       | 23.6%       |
| ResNet-50, FRN+TLU  | 24.0%       |

With cosine LR schedule:

| Model               | Top 1 Error |
|:--------------------|:------------|
| ResNet-50, BN       | 23.0%       |
| ResNet-50, FRN+TLU  | 23.2%       |

Experiments are only run once. Typical variance of such training is roughly ±0.1 around the mean.

### Comparison with the Paper
Results in the Filter Response Normalization paper uses a different recipe. Potential differences include:

1. Input image size is 299x299 v.s. our 224x224.
1. The use of "ResNet-v2" v.s. our classic ResNet. In our ResNet, activation do not always come
   immediately after normalization, which may affect the use of TLU.
1. Training length is unclear (seems to be "300k steps" with batch size 256) v.s. our 100 epochs.
1. Input augmentation may be different.
1. Exact definition of cosine LR schedule may be different.

The paper reports the following results (with cosine LR schedule):

| Model                | Top 1 Error |
|:---------------------|:------------|
| ResNetV2-50, BN      | 23.8%       |
| ResNetV2-50, FRN+TLU | 22.8%       |
