from super_image import Trainer, TrainingArguments, RnanConfig, RnanModel
from super_image.data import EvalDataset, TrainAugmentDataset


def train_model(train_file, eval_file, scale, output_dir):
    train_dataset = TrainAugmentDataset(train_file, scale=scale)
    val_dataset = EvalDataset(eval_file)

    training_args = TrainingArguments(
        output_dir=output_dir,                  # output directory
        num_train_epochs=1000,                  # total number of training epochs
    )

    config = RnanConfig(
        scale=scale,                                # train a model to upscale 4x
    )
    model = RnanModel(config)

    trainer = Trainer(
        model=model,                            # the instantiated model to be trained
        args=training_args,                     # training arguments, defined above
        train_dataset=train_dataset,            # training dataset
        eval_dataset=val_dataset                # evaluation dataset
    )

    trainer.train()


train_model('../BAM/DIV2K_train_HR_x4_train.h5', '../BAM/DIV2K_val_HR_x4_val.h5', 4, './results_rnan_4x')
train_model('../BAM/DIV2K_train_HR_x2_train.h5', '../BAM/DIV2K_val_HR_x2_val.h5', 2, './results_rnan_2x')
train_model('../BAM/DIV2K_train_HR_x3_train.h5', '../BAM/DIV2K_val_HR_x3_val.h5', 3, './results_rnan_3x')
