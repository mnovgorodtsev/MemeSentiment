from classic_analysis.resnet_bert_fusion.late_fusion.model import LateFusionTrainer

if __name__ == "__main__":

    hyperparams = [
        {"w_image": round(w / 10, 1), "w_text": round(1 - w / 10, 1)}
        for w in range(1, 10)
    ]

    trainer = LateFusionTrainer(
        resnet_path="./models/resnet_multitask_model_attention_final/model_weights.pt",
        bert_path="./models/bert_multitask_model_mlp_deep_final/model_weights.pt",
        results_path="./results/late_fusion/training_results_late_fusion.csv",
        use_mlflow=False,
    )

    best_params, best_acc = trainer.train(hyperparams=hyperparams)
    print("Best hyparams", best_params)
