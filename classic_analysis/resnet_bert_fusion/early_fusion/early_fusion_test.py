from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer

if __name__ == "__main__":
    trainer = EarlyFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        save_path="./models/early_fusion_model/model_weights.pt",
        use_mlflow=True,
    )

    trainer.test()
