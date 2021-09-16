from datasets import load_dataset
from super_image import EdsrModel, MsrnModel, A2nModel, PanModel, CarnModel, MdsrModel, AwsrnModel, HanModel, \
    DrlnModel, RcanModel
from super_image.data import EvalDataset, EvalMetrics


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    dataset_names = ['eugenesiow/Set5', 'eugenesiow/Set14', 'eugenesiow/BSD100', 'eugenesiow/Urban100']
    # dataset_names = ['eugenesiow/Urban100']
    scales = [4]
    for scale in scales:
        for dataset_name in dataset_names:
            # dataset = dataset.map(map_to_array)
            dataset = load_dataset(dataset_name, f'bicubic_x{scale}', split='validation')
            # dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
            # dataset = load_dataset('eugenesiow/PIRM', 'bicubic_x2', split='test')
            eval_dataset = EvalDataset(dataset)
            # model = A2nModel.from_pretrained('eugenesiow/a2n', scale=scale)
            # model = A2nModel.from_pretrained('../../super-image-models/a2n', scale=scale)
            # model = CarnModel.from_pretrained('../../super-image-models/carn', scale=scale)
            # model = PanModel.from_pretrained('../../super-image-models/pan', scale=scale)
            # model = MdsrModel.from_pretrained('../../super-image-models/mdsr', scale=scale)
            # model = AwsrnModel.from_pretrained('../../super-image-models/awsrn', scale=scale)
            # model = HanModel.from_pretrained('../../super-image-models/han', scale=scale)
            model = DrlnModel.from_pretrained('../../super-image-models/drln', scale=scale)
            # model = RcanModel.from_pretrained('../../super-image-models/rcan-bam', scale=scale)
            # model = MsrnModel.from_pretrained('../../super-image-models/msrn', scale=scale)
            # model = EdsrModel.from_pretrained('../../super-image-models/edsr', scale=scale)
            # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
            # model = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=scale)
            # model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=scale)
            print(count_parameters(model))
            EvalMetrics().evaluate(model, eval_dataset)
