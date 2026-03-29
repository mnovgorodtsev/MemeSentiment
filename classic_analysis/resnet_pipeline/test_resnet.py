from classic_analysis.base import MultiTaskTrainer
from classic_analysis.resnet_pipeline.model import ResNetMultiTaskModel

if __name__ == "__main__":
    model = ResNetMultiTaskModel()

    trainer = MultiTaskTrainer(
        model=model,
        csv_path="data/memotion_dataset_7k/labels.csv",
        data_type="image",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/resnet_multitask_model",
        test=True,
    )

    trainer.test()