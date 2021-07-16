<h1 align="center">super-image</h1>

<p align="center">
    <a href="https://eugenesiow.github.io/super-image/">
        <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat">
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

With [`pipx`](https://github.com/pipxproject/pipx):
```bash
python3.6 -m pip install --user pipx

pipx install --python python3.6 super-image
```

## Quick Start

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