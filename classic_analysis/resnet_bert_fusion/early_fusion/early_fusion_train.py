from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer
from utils.read_config import load_hyperparams


if __name__ == "__main__":
    hyperparams = load_hyperparams("early.json")

    # GRID SEARCH PART
    trainer = EarlyFusionTrainer(
        save_path="./models/early_fusion_model/model_weights.pt",
        results_path="./results/early_fusion/grid_search_results.csv",
        use_mlflow=False,
    )

    best_params, best_acc = trainer.train(hyperparams=hyperparams)

    # TRAIN PART
    trainer_final = EarlyFusionTrainer(
        save_path="./models/early_fusion_model/model_weights_final.pt",
        results_path="./results/early_fusion/training_results_final.csv",
        use_mlflow=False,
    )
 
    trainer_final.train(hyperparams=[best_params])
