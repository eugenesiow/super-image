# Evaluation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Evaluate_Pretrained_super_image_Models.ipynb "Open in Colab")

Evaluate pretrained `super-image` models with common image super resolution datasets.

## Setting up the Environment

#### Install the library

We will install the `super-image` and huggingface `datasets` library using `pip install`.

```bash
pip install -qq datasets super-image
```

## Loading the dataset
We download the [`Set5`](https://huggingface.co/datasets/eugenesiow/Set5) dataset using the huggingface `datasets` library. 

- Note that you can change `bicubic_x2` to any of [`bicubic_x2`, `bicubic_x3` or `bicubic_x4`].
- You can also explore more super resolution datasets [here](https://huggingface.co/datasets?filter=task_ids:other-other-image-super-resolution).

```python
from datasets import load_dataset

dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
```

If you want to preview the first image (high resolution and the low resolution (half sized) images) from the dataset.

```python
import cv2

cv2.imshow(cv2.imread(dataset[0]["hr"]))
cv2.imshow(cv2.imread(dataset[0]["lr"]))
```

## Evaluating the Model (Running Inference)

To evaluate the a model for the [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Quality_estimation_with_PSNR) and [SSIM](https://en.wikipedia.org/wiki/Structural_similarity#Algorithm) metrics we run the following code:

* `EvalDataset(dataset)` converts the dataset to an evaluation dataset that can be fed in to a PyTorch dataloader.
* `EdsrModel.from_pretrained` - Download and load a small, pre-trained deep-learning model to the `model` variable. You can replace this with [other](https://huggingface.co/models?filter=super-image) pretrained models.
* `EvalMetrics().evaluate(model, eval_dataset)` - Run the evaluation on the `eval_dataset` using the `model`.

```python
from super_image import EdsrModel
from super_image.data import EvalDataset, EvalMetrics

eval_dataset = EvalDataset(dataset)
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
EvalMetrics().evaluate(model, eval_dataset)
```

We can see from the output that the PSNR for this model on this dataset is `38.02` and the SSIM is `0.9607`.

## Try Other Models and Datasets

- You can replace the `EdsrModel` with [other](https://huggingface.co/models?filter=super-image) pretrained models.
- You can replace the [`Set5`](https://huggingface.co/datasets/eugenesiow/Set5) dataset with other datasets [here](https://huggingface.co/datasets?filter=task_ids:other-other-image-super-resolution).
- You can try different scales: `bicubic_x2`, `bicubic_x3` or `bicubic_x4`
- Compare the performance via the [leaderboard](https://github.com/eugenesiow/super-image#scale-x2).