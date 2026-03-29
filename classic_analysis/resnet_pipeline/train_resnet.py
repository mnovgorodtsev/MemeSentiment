from classic_analysis.base import MultiTaskTrainer
from classic_analysis.resnet_pipeline.model import ResNetMultiTaskModel

if __name__ == "__main__":
    model = ResNetMultiTaskModel()

    hyperparams = [
        {"batch_size": 16, "epochs": 3, "learning_rate": 2e-5, "lr_finetune": 1e-4},
        {"batch_size": 32, "epochs": 3, "learning_rate": 2e-5, "lr_finetune": 1e-4},
        {"batch_size": 32, "epochs": 5, "learning_rate": 1e-5, "lr_finetune": 5e-5},
    ]

    trainer = MultiTaskTrainer(
        model=model,
        csv_path="data/memotion_dataset_7k/labels.csv",
        data_type="image",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/resnet_multitask_model",
        results_path="./results/resnet/training_results_resnet.csv",
        use_mlflow=True,
    )

    best_params, best_val_acc = trainer.train(hyperparams=hyperparams)
    print("Best hyparams", best_params)
