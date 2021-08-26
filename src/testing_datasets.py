from datasets import load_dataset
from super_image import EdsrModel, MsrnModel, A2nModel, PanModel, CarnModel, MdsrModel
from super_image.data import EvalDataset, EvalMetrics


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


# dataset = dataset.map(map_to_array)
dataset = load_dataset('eugenesiow/Set14', 'bicubic_x2', split='validation')
# dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
# dataset = load_dataset('eugenesiow/PIRM', 'bicubic_x2', split='test')
eval_dataset = EvalDataset(dataset)
# model = A2nModel.from_pretrained('eugenesiow/a2n', scale=2)
# model = A2nModel.from_pretrained('../../super-image-models/a2n', scale=3)
# model = CarnModel.from_pretrained('../../super-image-models/carn', scale=3)
# model = PanModel.from_pretrained('../../super-image-models/pan', scale=3)
model = MdsrModel.from_pretrained('../../super-image-models/mdsr', scale=2)
# model = MsrnModel.from_pretrained('../../super-image-models/msrn', scale=2)
print(count_parameters(model))
# model = EdsrModel.from_pretrained('./results', scale=2)
# model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
# model = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=2)
# model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)
EvalMetrics().evaluate(model, eval_dataset)
