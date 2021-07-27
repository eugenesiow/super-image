<h1 align="center">super-image</h1>

<p align="center">
    <a href="https://eugenesiow.github.io/super-image/">
        <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs-blue.svg?style=flat">
    </a>
    <a href="https://github.com/eugenesiow/super-image/blob/main/LICENSE">
		<img alt="GitHub" src="https://img.shields.io/github/license/eugenesiow/super-image.svg?color=blue">
	</a>
    <a href="https://pypi.org/project/super-image/">
        <img alt="pypi version" src="https://img.shields.io/pypi/v/super-image.svg">
    </a>
</p>

<h3 align="center">
    <p>State-of-the-art image super resolution models for PyTorch.</p>
</h3>


## Requirements

super-image requires Python 3.6 or above.

## Installation

With `pip`:
```bash
pip install super-image
```

## Quick Start

Quickly utilise pre-trained models for upscaling your images 2x, 3x and 4x. See the full list of models [below](#pre-trained-models).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Upscale_Images_with_Pretrained_super_image_Models.ipynb "Open in Colab")

```python
from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
image = Image.open(requests.get(url, stream=True).raw)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')
```

## Pre-trained Models
Pre-trained models are available at various scales and hosted at the awesome [`huggingface_hub`](https://huggingface.co/models?filter=super-image). By default the models were pretrained on [DIV2K](https://huggingface.co/datasets/eugenesiow/Div2k), a dataset of 800 high-quality (2K resolution) images for training, augmented to 4000 images and uses a dev set of 100 validation images (images numbered 801 to 900). 

The leaderboard below shows the 
[PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Quality_estimation_with_PSNR) / [SSIM](https://en.wikipedia.org/wiki/Structural_similarity#Algorithm) 
metrics for each model at various scales on various test sets ([Set5](https://huggingface.co/datasets/eugenesiow/Set5), 
[Set14](https://huggingface.co/datasets/eugenesiow/Set14), 
[BSD100](https://huggingface.co/datasets/eugenesiow/BSD100), 
[Urban100](https://huggingface.co/datasets/eugenesiow/Urban100)). The **higher the better**. 
All training was to 1000 epochs (some publications, like a2n, train to >1000 epochs in their experiments). 

### Scale x2

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |**38.02/0.9608**   |**33.73/0.9186**  	|**33.78/0.9253**   |**32.08/0.9276**   |
|2      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |38.02/0.9607       |33.66/0.9180       |33.77/0.9254       |32.04/0.9276       |
|3      |[a2n](https://huggingface.co/eugenesiow/a2n)   	        |1.0m           |37.87/0.9602       |33.54/0.9171       |33.67/0.9244       |31.71/0.9240       |

### Scale x3

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |**35.13/0.9408**   |**31.06/0.8588**  	|**29.65/0.8196**   |**29.26/0.8736**   |
|2      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |35.01/0.9402       |31.01/0.8583       |29.63/0.8190       |29.19/0.8722       |

### Scale x4

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[msrn](https://huggingface.co/eugenesiow/msrn)             |6.1m           |32.19/0.8951       |**28.78/0.7862**   |**28.53/0.7657**   |**26.12/0.7866**   |
|2      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |**32.26/0.8955**   |28.78/0.7859       |28.51/0.7651       |26.10/0.7857       |
|3      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |32.12/0.8947       |28.72/0.7845       |28.50/0.7644       |26.02/0.7832       |
|4      |[a2n](https://huggingface.co/eugenesiow/a2n)               |1.0m           |32.07/0.8933       |28.68/0.7830       |28.44/0.7624       |25.89/0.7787       |