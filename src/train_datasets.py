# import torch
from datasets import load_dataset
from super_image import EdsrModel
from super_image.data import EvalDataset, EvalMetrics


# dataset = dataset.map(map_to_array)
dataset = load_dataset('eugenesiow/Set5', 'bicubic_x3', split='validation')
eval_dataset = EvalDataset(dataset)
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=3)
EvalMetrics().evaluate(model, eval_dataset)
