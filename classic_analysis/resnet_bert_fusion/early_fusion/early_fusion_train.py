from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer


if __name__ == "__main__":
    hyperparams = [
        {"batch_size": 16, "epochs": 3, "learning_rate": 2e-5, "lr_finetune": 1e-4},
        {"batch_size": 32, "epochs": 3, "learning_rate": 2e-5, "lr_finetune": 1e-4},
        {"batch_size": 32, "epochs": 5, "learning_rate": 1e-5, "lr_finetune": 5e-5},
    ]

    trainer = EarlyFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/early_fusion_model/model_weights.pt",
        use_mlflow=True,
    )

    best_params, best_acc = trainer.train(hyperparams=hyperparams)
    print("Best params:", best_params)