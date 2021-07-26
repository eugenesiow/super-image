from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ..utils.metrics import AverageMeter, calc_psnr, calc_ssim, convert_rgb_to_y, denormalize, get_scale_from_dataset


class EvalMetrics:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: nn.Module, dataset: Dataset, scale: int = None):
        if scale is None:
            if len(dataset) > 0:
                scale = get_scale_from_dataset(dataset)
            else:
                raise ValueError(f"Unable to calculate scale from empty dataset.")

        eval_dataloader = DataLoader(dataset=dataset, batch_size=1)
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        for i, data in tqdm(enumerate(eval_dataloader), total=len(dataset)):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                preds = model(inputs)

            preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
            labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

            preds = preds[scale:-scale, scale:-scale]
            labels = labels[scale:-scale, scale:-scale]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))
        print('scale:{}     eval psnr: {:.6f}   ssim: {:.6f}'.format(str(scale), epoch_psnr.avg, epoch_ssim.avg))
