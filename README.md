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
|4      |[carn-bam](https://huggingface.co/eugenesiow/carn-bam)     |1.6m           |37.83/0.96         |33.51/0.9166       |33.64/0.924        |31.53/0.922        |
|5      |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |260k           |37.7/0.9596        |33.4/0.9161        |33.6/0.9234        |31.35/0.92         |

### Scale x3

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[msrn](https://huggingface.co/eugenesiow/msrn)             |6.1m           |35.12/0.9409       |**31.08/0.8593**   |**29.67/0.8198**   |**29.31/0.8743**   |
|2      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |**35.13/0.9408**   |31.06/0.8588  	    |29.65/0.8196       |29.26/0.8736       |
|3      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |35.01/0.9402       |31.01/0.8583       |29.63/0.8190       |29.19/0.8722       |
|4      |[a2n](https://huggingface.co/eugenesiow/a2n)   	        |1.0m           |34.8/0.9387        |30.94/0.8568       |29.56/0.8173       |28.95/0.8671       |
|5      |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |260k           |34.62/0.9371       |30.83/0.8545       |29.47/0.8153       |28.64/0.861        |

### Scale x4

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[msrn](https://huggingface.co/eugenesiow/msrn)             |6.1m           |32.19/0.8951       |**28.78/0.7862**   |**28.53/0.7657**   |**26.12/0.7866**   |
|2      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |**32.26/0.8955**   |28.78/0.7859       |28.51/0.7651       |26.10/0.7857       |
|3      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |32.12/0.8947       |28.72/0.7845       |28.50/0.7644       |26.02/0.7832       |
|4      |[a2n](https://huggingface.co/eugenesiow/a2n)               |1.0m           |32.07/0.8933       |28.68/0.7830       |28.44/0.7624       |25.89/0.7787       |
|5      |[carn-bam](https://huggingface.co/eugenesiow/carn-bam)     |1.6m           |32.0/0.8923        |28.62/0.7822       |28.41/0.7614       |25.77/0.7741       |
|6      |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |270k           |31.9/0.8911        |28.54/0.7795       |28.32/0.7591       |25.6/0.7691        |

You can find a notebook to easily run evaluation on pretrained models below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Evaluate_Pretrained_super_image_Models.ipynb "Open in Colab")

## Train Models

We need the huggingface [datasets](https://huggingface.co/datasets?filter=task_ids:other-other-image-super-resolution) library to download the data:
```bash
pip install datasets
```
The following code gets the data and preprocesses/augments the data.

```python
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop

augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")                                # download and augment the data with the five_crop method
train_dataset = TrainDataset(augmented_dataset)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation'))      # prepare the eval dataset for the PyTorch DataLoader
```

The training code is provided below:
```python
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig

training_args = TrainingArguments(
    output_dir='./results',                 # output directory
    num_train_epochs=1000,                  # total number of training epochs
)

config = EdsrConfig(
    scale=4,                                # train a model to upscale 4x
)
model = EdsrModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Train_super_image_Models.ipynb "Open in Colab")