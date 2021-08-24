from super_image import Trainer, TrainingArguments, DrnConfig, DrnModel
from super_image.data import EvalDataset, TrainAugmentDataset


def train_model(train_file, eval_file, scale, output_dir):
    train_dataset = TrainAugmentDataset(train_file, scale=scale)
    val_dataset = EvalDataset(eval_file)

    training_args = TrainingArguments(
        output_dir=output_dir,                  # output directory
        num_train_epochs=1000,                  # total number of training epochs
    )

    config = DrnConfig(
        scale=scale,                                # train a model to upscale 4x
        n_blocks=30,                                # DRN-S has B=30, F=16 for 4x
        n_feats=16,
        # n_blocks=40,                              # DRN-L has B=40, F=20 for 4x
        # n_feats=20,
    )
    model = DrnModel(config)

    trainer = Trainer(
        model=model,                            # the instantiated model to be trained
        args=training_args,                     # training arguments, defined above
        train_dataset=train_dataset,            # training dataset
        eval_dataset=val_dataset                # evaluation dataset
    )

    trainer.train()


train_model('../BAM/DIV2K_train_HR_x4_train.h5', '../BAM/DIV2K_val_HR_x4_val.h5', 4, './results_drn_s_4x')
