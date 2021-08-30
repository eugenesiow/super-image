# Lightweight Image Super-Resolution with Adaptive Weighted Learning Network (AWSRN)

## Overview

Deep learning has been successfully applied to the single-image super-resolution (SISR) task with great performance in recent years. However, most convolutional neural network based SR models require heavy computation, which limit their real-world applications. In this work, a lightweight SR network, named Adaptive Weighted Super-Resolution Network (AWSRN), is proposed for SISR to address this issue. A novel local fusion block (LFB) is designed in AWSRN for efficient residual learning, which consists of stacked adaptive weighted residual units (AWRU) and a local residual fusion unit (LRFU). Moreover, an adaptive weighted multi-scale (AWMS) module is proposed to make full use of features in reconstruction layer. AWMS consists of several different scale convolutions, and the redundancy scale branch can be removed according to the contribution of adaptive weights in AWMS for lightweight network. The experimental results on the commonly used datasets show that the proposed lightweight AWSRN achieves superior performance on ×2, ×3, ×4, and ×8 scale factors to state-of-the-art methods with similar parameters and computational overhead.The PAN model proposes a a lightweig1ht convolutional neural network for image super resolution. Pixel attention (PA) is similar to channel attention and spatial attention in formulation. PA however produces 3D attention maps instead of a 1D attention vector or a 2D map. This attention scheme introduces fewer additional parameters but generates better SR results.Deep learning has been successfully applied to the single-image super-resolution (SISR) task with great performance in recent years. However, most convolutional neural network based SR models require heavy computation, which limit their real-world applications. In this work, a lightweight SR network, named Adaptive Weighted Super-Resolution Network (AWSRN), is proposed for SISR to address this issue. A novel local fusion block (LFB) is designed in AWSRN for efficient residual learning, which consists of stacked adaptive weighted residual units (AWRU) and a local residual fusion unit (LRFU). Moreover, an adaptive weighted multi-scale (AWMS) module is proposed to make full use of features in reconstruction layer. AWMS consists of several different scale convolutions, and the redundancy scale branch can be removed according to the contribution of adaptive weights in AWMS for lightweight network. The experimental results on the commonly used datasets show that the proposed lightweight AWSRN achieves superior performance on ×2, ×3, ×4, and ×8 scale factors to state-of-the-art methods with similar parameters and computational overhead.

This model also applies the balanced attention (BAM) method invented by [Wang et al. (2021)](https://arxiv.org/abs/2104.07566) to further improve the results.

It was introduced in the paper [Lightweight Image Super-Resolution with Adaptive Weighted Learning Network](https://arxiv.org/abs/1904.02358) by Wang et al. (2019) and first released in [this repository](https://github.com/ChaofWang/AWSRN).   

## AwsrnConfig

::: super_image.AwsrnConfig

## AwsrnModel

::: super_image.AwsrnModel
