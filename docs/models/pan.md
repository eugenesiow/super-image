# Pixel Attention Network (PAN)

## Overview

The PAN model proposes a a lightweight convolutional neural network for image super resolution. Pixel attention (PA) is similar to channel attention and spatial attention in formulation. PA however produces 3D attention maps instead of a 1D attention vector or a 2D map. This attention scheme introduces fewer additional parameters but generates better SR results.

This model also applies the balanced attention (BAM) method invented by [Wang et al. (2021)](https://arxiv.org/abs/2104.07566) to further improve the results.

It was introduced in the paper  [Efficient Image Super-Resolution Using Pixel Attention](https://arxiv.org/abs/2010.01073) by Zhao et al. (2020) and first released in [this repository](https://github.com/zhaohengyuan1/PAN).  

## PanConfig

::: super_image.PanConfig

## PanModel

::: super_image.PanModel
