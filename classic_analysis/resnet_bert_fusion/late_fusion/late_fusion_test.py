from classic_analysis.resnet_bert_fusion.late_fusion.model import LateFusionTrainer

if __name__ == "__main__":

    trainer = LateFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        resnet_path="./models/resnet_multitask_model_linear_final/model_weights.pt",
        bert_path="./models/bert_multitask_model_mlp_deep_final/model_weights.pt",
        results_path="./results/late_fusion/training_results_late_fusion.csv",
        use_mlflow=False,
    )

    trainer.test(f"./results/test/late_fusion_model_final")
