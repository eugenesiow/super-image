from datasets import load_dataset
from super_image import EdsrModel, MsrnModel, A2nModel
from super_image.data import EvalDataset, EvalMetrics


# dataset = dataset.map(map_to_array)
# dataset = load_dataset('eugenesiow/Urban100', 'bicubic_x2', split='validation')
dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
eval_dataset = EvalDataset(dataset)
# model = A2nModel.from_pretrained('eugenesiow/a2n', scale=2)
model = EdsrModel.from_pretrained('./results', scale=2)
# model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
# model = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=2)
# model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)
EvalMetrics().evaluate(model, eval_dataset)
