from super_image import Trainer, TrainingArguments, EdsrConfig, EdsrModel
from super_image.data import EvalDataset, TrainAugmentDataset, DatasetBuilder

# DatasetBuilder.prepare(
#     base_path='../../super-image-datasets/DIV2K2017/DIV2K/DIV2K_train_HR',
#     output_path='../../super-image-datasets/div2k_4x_train.h5',
#     scale=4,
#     do_augmentation=True
# )
# DatasetBuilder.prepare(
#     base_path='../../super-image-datasets/SR_training_datasets/T91/',
#     output_path='../../super-image-datasets/t91_4x_train.h5',
#     scale=4,
#     do_augmentation=True
# )
# DatasetBuilder.prepare(
#     base_path='../../super-image-datasets/DIV2K2017/DIV2K/DIV2K_val_HR/',
#     output_path='../../super-image-datasets/div2k_4x_val.h5',
#     scale=4,
#     do_augmentation=True
# )
train_dataset = TrainAugmentDataset('../../super-image-datasets/div2k_4x_val.h5', scale=4)
# train_dataset = TrainAugmentDataset('../../super-image-datasets/div2k_4x_train.h5', scale=4)
val_dataset = EvalDataset('../../super-image-models/test/Set5_x4.h5')

training_args = TrainingArguments(
    output_dir='./results',                 # output directory
    num_train_epochs=1000,                  # total number of training epochs
)

config = EdsrConfig(
    scale=4,                                # train a model to upscale 4x
    supported_scales=[2, 3, 4],
)
model = EdsrModel(config)

# print(training_args)
# print(config.to_dict())

trainer = Trainer(
    model=model,                            # the instantiated model to be trained
    args=training_args,                     # training arguments, defined above
    train_dataset=train_dataset,            # training dataset
    eval_dataset=val_dataset                # evaluation dataset
)

trainer.train()
# trainer.save_model()
