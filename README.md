<h1 align="center">super-image</h1>

<p align="center">
    <a href="https://pypi.org/project/super-image/">
        <img alt="downloads" src="https://img.shields.io/pypi/dm/super-image">
    </a>
    <a href="https://eugenesiow.github.io/super-image/">
        <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs-blue.svg?style=flat">
    </a>
    <a href="https://github.com/eugenesiow/super-image/blob/main/LICENSE">
		<img alt="GitHub" src="https://img.shields.io/github/license/eugenesiow/super-image.svg?color=blue">
	</a>
    <a href="https://pypi.org/project/super-image/">
        <img alt="pypi version" src="https://img.shields.io/pypi/v/super-image.svg">
    </a>
    <a href="https://huggingface.co/spaces/eugenesiow/super-image">
        <img alt="demo app" src="https://img.shields.io/badge/demo-spaces-purple.svg?style=flat">
    </a>
</p>

<p align="center">
    <img align="center" alt="the super-image library's MSRN x4 model" src="https://github.com/eugenesiow/super-image/raw/main/docs/banner.png">
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
|1      |[drln-bam](https://huggingface.co/eugenesiow/drln-bam)     |34m            |**38.23/0.9614**   |33.95/0.9206  	    |**33.95/0.9269**   |**32.81/0.9339**   |
|2      |[edsr](https://huggingface.co/eugenesiow/edsr)  	        |41m            |38.19/0.9612       |**33.99/0.9215**  	|33.89/0.9266       |32.68/0.9331       |
|3      |[msrn](https://huggingface.co/eugenesiow/msrn)  	        |5.9m           |38.08/0.9609       |33.75/0.9183  	    |33.82/0.9258       |32.14/0.9287       |
|4      |[mdsr](https://huggingface.co/eugenesiow/mdsr)             |2.7m           |38.04/0.9608       |33.71/0.9184       |33.79/0.9256       |32.14/0.9283       |
|5      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |38.02/0.9608       |33.73/0.9186  	    |33.78/0.9253       |32.08/0.9276       |
|6      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |38.02/0.9607       |33.66/0.9180       |33.77/0.9254       |32.04/0.9276       |
|7      |[mdsr-bam](https://huggingface.co/eugenesiow/mdsr-bam)     |2.7m           |38/0.9607          |33.68/0.9182       |33.77/0.9253       |32.04/0.9272       |
|8      |[awsrn-bam](https://huggingface.co/eugenesiow/awsrn-bam)   |1.4m           |37.99/0.9606       |33.66/0.918        |33.76/0.9253       |31.95/0.9265       |
|9      |[a2n](https://huggingface.co/eugenesiow/a2n)   	        |1.0m           |37.87/0.9602       |33.54/0.9171       |33.67/0.9244       |31.71/0.9240       |
|10     |[carn](https://huggingface.co/eugenesiow/carn)             |1.6m           |37.89/0.9602       |33.53/0.9173       |33.66/0.9242       |31.62/0.9229       |
|11     |[carn-bam](https://huggingface.co/eugenesiow/carn-bam)     |1.6m           |37.83/0.96         |33.51/0.9166       |33.64/0.924        |31.53/0.922        |
|12     |[pan](https://huggingface.co/eugenesiow/pan)               |260k           |37.77/0.9599       |33.42/0.9162       |33.6/0.9235        |31.31/0.9197       |
|13     |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |260k           |37.7/0.9596        |33.4/0.9161        |33.6/0.9234        |31.35/0.92         |

### Scale x3

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[drln-bam](https://huggingface.co/eugenesiow/drln-bam)     |34m            |35.3/0.9422        |**31.27/0.8624**   |**29.78/0.8224**   |**29.82/0.8828**   |
|1      |[edsr](https://huggingface.co/eugenesiow/edsr)             |44m            |**35.31/0.9421**   |31.18/0.862        |29.77/0.8224       |29.75/0.8825       |
|1      |[msrn](https://huggingface.co/eugenesiow/msrn)             |6.1m           |35.12/0.9409       |31.08/0.8593       |29.67/0.8198       |29.31/0.8743       |
|2      |[mdsr](https://huggingface.co/eugenesiow/mdsr)  	        |2.9m           |35.11/0.9406       |31.06/0.8593  	    |29.66/0.8196       |29.29/0.8738       |
|3      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |35.13/0.9408       |31.06/0.8588  	    |29.65/0.8196       |29.26/0.8736       |
|4      |[mdsr-bam](https://huggingface.co/eugenesiow/mdsr-bam)  	|2.9m           |35.07/0.9402       |31.04/0.8582       |29.62/0.8188       |29.16/0.8717       |
|5      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |35.01/0.9402       |31.01/0.8583       |29.63/0.8190       |29.19/0.8722       |
|6      |[awsrn-bam](https://huggingface.co/eugenesiow/awsrn-bam)   |1.5m           |35.05/0.9403       |31.01/0.8581       |29.63/0.8188       |29.14/0.871        |
|7      |[carn](https://huggingface.co/eugenesiow/carn)             |1.6m           |34.88/0.9391       |30.93/0.8566       |29.56/0.8173       |28.95/0.867        |
|8      |[a2n](https://huggingface.co/eugenesiow/a2n)   	        |1.0m           |34.8/0.9387        |30.94/0.8568       |29.56/0.8173       |28.95/0.8671       |
|9      |[carn-bam](https://huggingface.co/eugenesiow/carn-bam)     |1.6m           |34.82/0.9385       |30.9/0.8558        |29.54/0.8166       |28.84/0.8648       |
|10     |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |260k           |34.62/0.9371       |30.83/0.8545       |29.47/0.8153       |28.64/0.861        |
|11     |[pan](https://huggingface.co/eugenesiow/pan)               |260k           |34.64/0.9376       |30.8/0.8544        |29.47/0.815        |28.61/0.8603       |

### Scale x4

|Rank   |Model  	                                                |Params         |Set5  	            |Set14  	        |BSD100  	        |Urban100  	        |
|---    |---	                                                    |---            |---                |---	            |---	            |---	            |
|1      |[drln-bam](https://huggingface.co/eugenesiow/drln-bam)     |34m            |32.49/0.8986       |**28.94/0.7899**   |**28.63/0.7686**   |26.53/0.7991       |
|2      |[edsr](https://huggingface.co/eugenesiow/edsr)             |43m            |**32.5/0.8986**    |28.92/0.7899       |28.62/0.7689       |**26.53/0.7995**   |
|3      |[msrn](https://huggingface.co/eugenesiow/msrn)             |6.1m           |32.19/0.8951       |28.78/0.7862       |28.53/0.7657       |26.12/0.7866       |
|4      |[msrn-bam](https://huggingface.co/eugenesiow/msrn-bam)  	|5.9m           |32.26/0.8955       |28.78/0.7859       |28.51/0.7651       |26.10/0.7857       |
|5      |[mdsr](https://huggingface.co/eugenesiow/mdsr)             |2.8m           |32.26/0.8953       |28.77/0.7856       |28.53/0.7653       |26.07/0.7851       |
|6      |[mdsr-bam](https://huggingface.co/eugenesiow/mdsr-bam)     |2.9m           |32.19/0.8949       |28.73/0.7847       |28.50/0.7645       |26.02/0.7834       |
|7      |[awsrn-bam](https://huggingface.co/eugenesiow/awsrn-bam)   |1.6m           |32.13/0.8947       |28.75/0.7851       |28.51/0.7647       |26.03/0.7838       |
|8      |[edsr-base](https://huggingface.co/eugenesiow/edsr-base)  	|1.5m           |32.12/0.8947       |28.72/0.7845       |28.50/0.7644       |26.02/0.7832       |
|9      |[a2n](https://huggingface.co/eugenesiow/a2n)               |1.0m           |32.07/0.8933       |28.68/0.7830       |28.44/0.7624       |25.89/0.7787       |
|10     |[carn](https://huggingface.co/eugenesiow/carn)             |1.6m           |32.05/0.8931       |28.67/0.7828       |28.44/0.7625       |25.85/0.7768       |
|11     |[carn-bam](https://huggingface.co/eugenesiow/carn-bam)     |1.6m           |32.0/0.8923        |28.62/0.7822       |28.41/0.7614       |25.77/0.7741       |
|12     |[pan](https://huggingface.co/eugenesiow/pan)               |270k           |31.92/0.8915       |28.57/0.7802       |28.35/0.7595       |25.63/0.7692       |
|13     |[pan-bam](https://huggingface.co/eugenesiow/pan-bam)       |270k           |31.9/0.8911        |28.54/0.7795       |28.32/0.7591       |25.6/0.7691        |
|14     |[han](https://huggingface.co/eugenesiow/han)               |16m            |31.21/0.8778       |28.18/0.7712       |28.09/0.7533       |25.1/0.7497        |
|15     |[rcan-bam](https://huggingface.co/eugenesiow/rcan-bam)     |15m            |30.8/0.8701        |27.91/0.7648       |27.91/0.7477       |24.75/0.7346       |

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