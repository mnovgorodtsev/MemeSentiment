from classic_analysis.base import MultiTaskTrainer
from classic_analysis.resnet_pipeline.model import (
    ResNetLinear,
    ResNetAttention,
    ResNetAdaptivePooling,
)
from utils.read_config import load_hyperparams

if __name__ == "__main__":
    model = ResNetAdaptivePooling()

    hyperparams = load_hyperparams("resnet_params.json")

    trainer = MultiTaskTrainer(
        model=model,
        csv_path="data/memotion_dataset_7k/labels.csv",
        data_type="image",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/resnet_multitask_model_pooling",
        results_path="./results/resnet/training_results_resnet_pooling.csv",
        use_mlflow=False,
    )

    best_params, best_val_acc = trainer.train(hyperparams=hyperparams)
    print("Best hyparams", best_params)
