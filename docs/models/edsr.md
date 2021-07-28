# Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)

## Overview

EDSR is a model that uses both deeper and wider architecture (32 ResBlocks and 256 channels) to improve performance. 
It uses both global and local skip connections, and up-scaling is done at the end of the network. 
It doesn't use batch normalization layers (input and output have similar distributions, normalizing intermediate 
features may not be desirable) instead it uses constant scaling layers to ensure stable training. 
An L1 loss function (absolute error) is used instead of L2 (MSE), the authors showed better performance empirically 
and it requires less computation.

The default parameters are for the base model (~5mb vs ~100mb) that includes just 16 ResBlocks and 64 channels.

It was introduced in the paper [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) by Lim et al. (2017) and first released in [this repository](https://github.com/sanghyun-son/EDSR-PyTorch). 

## EdsrConfig

::: super_image.EdsrConfig

## EdsrModel

::: super_image.EdsrModel
