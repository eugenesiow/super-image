# Attention in Attention Network for Image Super-Resolution (A2N)

## Overview

The A2N model proposes an attention in attention network (A2N) for highly accurate image SR. Specifically, the A2N consists of a non-attention branch and a coupling attention branch. Attention dropout module is proposed to generate dynamic attention weights for these two branches based on input features that can suppress unwanted attention adjustments. This allows attention modules to specialize to beneficial examples without otherwise penalties and thus greatly improve the capacity of the attention network with little parameter overhead. 

More importantly the model is lightweight and fast to train (~1.5m parameters, ~4mb).

It was introduced in the paper [Attention in Attention Network for Image Super-Resolution](https://arxiv.org/abs/2104.09497) by Chen et al. (2021) and first released in [this repository](https://github.com/haoyuc/A2N). 

## A2nConfig

::: super_image.A2nConfig

## A2nModel

::: super_image.A2nModel
