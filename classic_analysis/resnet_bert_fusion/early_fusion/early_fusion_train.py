from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer
from utils.read_config import load_hyperparams

if __name__ == "__main__":
    hyperparams = load_hyperparams("early.json")

    trainer = EarlyFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/early_fusion_model/model_weights.pt",
        use_mlflow=False,
    )

    best_params, best_acc = trainer.train(hyperparams=hyperparams)
    print("Best params:", best_params)
