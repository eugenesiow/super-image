# Prediction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Upscale_Images_with_Pretrained_super_image_Models.ipynb "Open in Colab")

Use the `super-image` library to quickly upscale an image.

## Setting up the Environment

#### Install the library

We will install the `super-image` using `pip install`.

```bash
pip install -qq super-image
```

## Load a Pretrained Model for Inference

Next we run a few lines of code to:

* `Image.open` and `requests.get` - Download an image from a URL (website) and store this as the `image` variable.
* `EdsrModel.from_pretrained` - Download and load a small, pre-trained deep-learning model to the `model` variable.
* `ImageLoader.load_image` - Load the image into the `model` using the `ImageLoader` helper.
* Use the model to run inference on the image (`inputs`).
* `ImageLoader.save_image` - Save the upscaled image output as a `.png` file using the `ImageLoader` helper.
* `ImageLoader.save_compare` - Save a `.png` that compares our upscaled image from the model with a baseline image using `Bicubic` upscaling.

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

View the comparison image to see, visually, how our model performed (on the right) against the baseline bicubic method (left).

```python
import cv2

img = cv2.imread('./scaled_2x_compare.png') 
cv2.imshow(img)
```

We can view the original image that we pulled from the URL/website using `cv2.imshow`.

```python
import numpy as np

cv2.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
```

## Try Other Models

- You can replace the `EdsrModel` with [other](https://huggingface.co/models?filter=super-image) pretrained models.
- You can try different scales: `bicubic_x2`, `bicubic_x3` or `bicubic_x4`
- Compare the performance via the [leaderboard](https://github.com/eugenesiow/super-image#scale-x2).