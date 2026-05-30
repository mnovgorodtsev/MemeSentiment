from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer

if __name__ == "__main__":
    trainer = EarlyFusionTrainer(
        save_path="./models/early_fusion_model/model_weights.pt",
        use_mlflow=False,
    )

    trainer.test(f"./results/test/early_fusion_model_final")
