import torch
import cv2
import numpy as np
from PIL import Image
from super_image.models import EdsrModel, EdsrConfig, MsrnModel, A2nConfig, A2nModel, PanModel, CarnModel, MdsrModel
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as nnf
import torchvision.transforms.functional as F
from super_image.data import EvalDatasetH5
from super_image.utils.metrics import AverageMeter, calc_psnr, calc_ssim, convert_rgb_to_y, denormalize


def evaluate_metrics(eval_file,  scale):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eval_dataset = EvalDatasetH5(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    opt = EdsrConfig(name_or_path='eugenesiow/edsr', scale=scale, n_resblocks=16, n_feats=64, n_colors=3,
                     rgb_range=255, res_scale=1, data_parallel=True)
    opt.to_json_file('../../../super-image-models/edsr/config_4x.json')
    print(opt.to_dict())
    # model = EdsrModel(opt).cuda()
    model = EdsrModel(opt)
    # model = nn.DataParallel(model).cuda()
    if opt.data_parallel and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    checkpoint = torch.load('../../../super-image-models/edsr/pytorch_model_4x.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()
    for i, data in enumerate(eval_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
            # print(preds[0].shape)

        preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
        labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

        preds = preds[scale:-scale, scale:-scale]
        labels = labels[scale:-scale, scale:-scale]

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

        # pred = preds.data.cpu().numpy()
        #
        # pred = pred[0].transpose((1, 2, 0)) * 255.0
        # pred = pred[scale:-scale, scale:-scale, :]
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('../../../super-image-models/' + str(i) + '.png', pred)

    print('scale:{}     eval psnr: {:.6f}   ssim: {:.6f}'.format(str(scale), epoch_psnr.avg, epoch_ssim.avg))


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained(input_dir, eval_file, scale, model_type='edsr'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    if model_type == 'edsr':
        model = EdsrModel.from_pretrained(input_dir, scale=scale)
    elif model_type == 'a2n':
        model = A2nModel.from_pretrained(input_dir, scale=scale)
    else:
        model = MsrnModel.from_pretrained(input_dir, scale=scale)
    print(f'params: {count_parameters(model)}')
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()
    for i, data in enumerate(eval_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
            # print(preds[0].shape)

        # print(type(preds), type(labels))

        preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
        labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

        preds = preds[scale:-scale, scale:-scale]
        labels = labels[scale:-scale, scale:-scale]

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        epoch_ssim.update(calc_ssim(preds, labels), len(inputs))
    print('scale:{}     eval psnr: {:.6f}   ssim: {:.6f}'.format(str(scale), epoch_psnr.avg, epoch_ssim.avg))


def output_image(input_dir, eval_file, scale, model_type='edsr'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eval_dataset = EvalDatasetH5(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    if model_type == 'edsr':
        model = EdsrModel.from_pretrained(input_dir, scale=scale)
    elif model_type == 'a2n':
        model = A2nModel.from_pretrained(input_dir, scale=scale)
    elif model_type == 'pan':
        model = PanModel.from_pretrained(input_dir, scale=scale)
    elif model_type == 'carn':
        model = CarnModel.from_pretrained(input_dir, scale=scale)
    elif model_type == 'mdsr':
        model = MdsrModel.from_pretrained(input_dir, scale=scale)
    else:
        model = MsrnModel.from_pretrained(input_dir, scale=scale)
    for i, data in enumerate(eval_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
        pred = preds.data.cpu().numpy()

        pred = pred[0].transpose((1, 2, 0)) * 255.0
        pred = pred[scale:-scale, scale:-scale, :]
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(cv2.cvtColor(inputs.numpy()[0].transpose((1, 2, 0)) * 255.0, cv2.COLOR_BGR2RGB),
                                (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
        hr_img = cv2.resize(cv2.cvtColor(labels.numpy()[0].transpose((1, 2, 0)) * 255.0, cv2.COLOR_BGR2RGB),
                            (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
        # im_bicubic = nnf.interpolate(cv2.cvtColor(inputs.numpy()[0].transpose((1, 2, 0)) * 255.0, cv2.COLOR_BGR2RGB),
        #                              size=(pred.shape[0], pred.shape[1]), mode='bicubic',
        #                              align_corners=False)
        # im_bicubic = F.resize(inputs, pred.shape[0])
        # im_bicubic = cv2.cvtColor(im_bicubic.transpose((1, 2, 0)) * 255.0, cv2.COLOR_BGR2RGB)
        # final = torch.cat([im_bicubic, torch.Tensor(pred)])
        # save_image(final, '../../../super-image-models/' + str(i) + '_compare.png', nrow=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCorner = (10, 30)
        bottomLeftCornerOfText = (10, pred.shape[0]-10)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(img=hr_img, text='x' + str(scale), org=topLeftCorner,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA,
                    thickness=4)
        cv2.putText(img=hr_img, text='x' + str(scale), org=topLeftCorner,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=[255, 255, 255], lineType=cv2.LINE_AA,
                    thickness=2)
        cv2.putText(hr_img, 'HR',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(img_resize, 'Bicubic',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(pred, model_type.upper(),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        numpy_horizontal = np.hstack((hr_img, img_resize, pred))
        cv2.imwrite('../../../super-image-models/results/' + model_type + '_' + str(i) + '_' + str(scale)
                    + '_compare.png', numpy_horizontal)
        # cv2.imwrite('../../../super-image-models/' + str(i) + '.png', img_resize)
        # cv2.imwrite('../../../super-image-models/' + str(i) + '_pred.png', pred)


# load_pretrained('../../../super-image-models/edsr', '../../../super-image-models/test/Set5_x4.h5', 4)
# load_pretrained('../../../super-image-models/edsr-mini', '../../../super-image-models/test/Set5_x4.h5', 4)
# load_pretrained('../../../super-image-models/edsr-mini', '../../../super-image-models/test/Set5_x2.h5', 2)
# load_pretrained('eugenesiow/edsr-base', '../../../super-image-models/test/Set5_x2.h5', 2)
# load_pretrained('../../../super-image-models/edsr-base', '../../../super-image-models/test/Set5_x3.h5', 3)
# load_pretrained('../../../super-image-models/edsr-base', '../../../super-image-models/test/Urban100_x3.h5', 3)
# load_pretrained('../../../super-image-models/edsr-base', '../../../super-image-models/test/BSD100_x2.h5', 2)
# load_pretrained('../../../super-image-models/edsr-base', '../../../super-image-models/test/Urban100_x2.h5', 2)
# output_image('eugenesiow/edsr-base', '../../../super-image-models/test/Set5_x2.h5', 2)
# output_image('eugenesiow/edsr-base', '../../../super-image-models/test/Set5_x4.h5', 4)
# output_image('eugenesiow/a2n', '../../../super-image-models/test/Set5_x4.h5', 4, model_type='a2n')
output_image('../../../super-image-models/mdsr-bam', '../../../super-image-models/test/Set5_x4.h5', 4, model_type='mdsr')
# output_image('../../../super-image-models/msrn-bam', '../../../super-image-models/test/Set5_x4.h5', 4, model_type='msrn')
# load_pretrained('eugenesiow/a2n', '../../../super-image-models/test/Set14_x4.h5', 4, model_type='a2n')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/BSD100_x4.h5', 4, model_type='a2n')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/Urban100_x4.h5', 4, model_type='a2n')
# load_pretrained('../../../super-image-models/msrn', '../../../super-image-models/test/Set14_x4.h5', 4, model_type='msrn')
# load_pretrained('../../../super-image-models/msrn', '../../../super-image-models/test/BSD100_x4.h5', 4, model_type='msrn')
# load_pretrained('../../../super-image-models/msrn', '../../../super-image-models/test/Urban100_x4.h5', 4, model_type='msrn')
# load_pretrained('eugenesiow/msrn-bam', '../../../super-image-models/test/Set5_x2.h5', 2, model_type='msrn')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/Set5_x2.h5', 2, model_type='a2n')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/Set14_x2.h5', 2, model_type='a2n')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/BSD100_x2.h5', 2, model_type='a2n')
# load_pretrained('../../../super-image-models/a2n', '../../../super-image-models/test/Urban100_x2.h5', 2, model_type='a2n')
# load_pretrained('../../../super-image-models/msrn-bam', '../../../super-image-models/test/Urban100_x3.h5', 3, model_type='msrn')
# load_pretrained('../../../super-image-models/edsr', '../../../super-image-models/test/DIV2K_val_HR_x4_val.h5', 4)
# load_pretrained('../../../super-image-models/msrn', '../../../super-image-models/test/Set5_x4.h5', 4)
# evaluate_metrics('../../../super-image-models/test/Set5_x4.h5', 4)
# evaluate_metrics('../../../super-image-models/test/BSD100_x4.h5', 4)
# evaluate_metrics('../../../super-image-models/test/Urban100_x4.h5', 4)
# evaluate_metrics('../../../super-image-models/test/Set14_x4.h5', 4)
