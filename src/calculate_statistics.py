from datasets import load_dataset

from super_image.utils.metrics import calculate_mean_std
from super_image.data import EvalDataset

# calculate_mean_std(EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train')))
calculate_mean_std(EvalDataset(load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')))
