# Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eugenesiow/super-image-notebooks/blob/master/notebooks/Train_super_image_Models.ipynb "Open in Colab")

Train `super-image` models for image super resolution tasks.

## Setting up the Environment

#### Install the library

We will install the `super-image` and huggingface `datasets` library using `pip install`.

```bash
pip install -qq datasets super-image
```

## Loading and Augmenting the Dataset
We download the [`Div2k`](https://huggingface.co/datasets/eugenesiow/Div2k) dataset using the huggingface `datasets` library. You can explore more super resolution datasets [here](https://huggingface.co/datasets?filter=task_ids:other-other-image-super-resolution). 

We then follow the pre-processing and augmentation method of [Wang et al. (2021)](https://arxiv.org/abs/2104.07566). This will take awhile, go grab a coffee.

- Note that you can change `bicubic_x4` to any of [`bicubic_x2`, `bicubic_x3` or `bicubic_x4`].
- If you don't want to do augmentation to your dataset, you can just do: `train_dataset = TrainDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train'))`
- If you want eval to be faster you can use the much smaller [Set5](https://huggingface.co/datasets/eugenesiow/Set5): `eval_dataset = EvalDataset(load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation'))`

```python
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop

augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='train')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")                                # download and augment the data with the five_crop method
train_dataset = TrainDataset(augmented_dataset)                                                     # prepare the train dataset for loading PyTorch DataLoader
eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation'))      # prepare the eval dataset for the PyTorch DataLoader
```

## Training the Model

We then train the model. It's best if you have a GPU.

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

We see that after each epoch of training, the PSNR and SSIM scores of the epoch on the validation set is reported.

The best model after 1000 epochs is saved.

## Try Other Architectures

- You can try the other [architectures](https://eugenesiow.github.io/super-image/models/edsr/) in `super-image`.
- Compare the performance via the [leaderboard](https://github.com/eugenesiow/super-image#scale-x2).
- View the various pretrained models on [huggingface hub](https://huggingface.co/models?filter=super-image).

Here is an example on another architecture, MSRN:

```python
from super_image import Trainer, TrainingArguments, MsrnModel, MsrnConfig

training_args = TrainingArguments(
    output_dir='./results_msrn',         # output directory
    num_train_epochs=2,                  # total number of training epochs
)

config = MsrnConfig(
    scale=4,                                # train a model to upscale 4x
    bam=True,                               # use balanced attention
)
model = MsrnModel(config)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

trainer.train()

```