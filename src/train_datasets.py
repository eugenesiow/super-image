from datasets import load_dataset
from super_image import Trainer, TrainingArguments, CarnModel, CarnConfig, EdsrModel, EdsrConfig
from super_image.data import EvalDataset, TrainDataset, augment_five_crop


# augmented_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='train')\
#     .map(augment_five_crop, batched=True, desc="Augmenting Dataset")
augmented_dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')\
    .map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset, patch_size=32)
# train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation'))
# eval_dataset = EvalDataset(load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation'))
training_args = TrainingArguments(
    output_dir='./results',                 # output directory
    num_train_epochs=1000,                  # total number of training epochs
)

# config = CarnConfig(
#     scale=2,                                # train a model to upscale 4x
#     bam=True,
# )
# model = CarnModel(config)

config = EdsrConfig(
    scale=2,                                # train a model to upscale 4x
)
model = EdsrModel(config)


trainer = Trainer(
    model=model,                            # the instantiated model to be trained
    args=training_args,                     # training arguments, defined above
    train_dataset=train_dataset,            # training dataset
    eval_dataset=eval_dataset               # evaluation dataset
)

trainer.train()
