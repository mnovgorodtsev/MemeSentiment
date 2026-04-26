from classic_analysis.resnet_pipeline.model import (
    ResNetLinear,
    ResNetAttention,
    ResNetAdaptivePooling,
)
from classic_analysis.base import initialize_model
from utils.read_config import load_hyperparams


if __name__ == "__main__":
    for model, prefix in zip(
        [ResNetLinear, ResNetAttention, ResNetAdaptivePooling], ["_linear", "_attention", "_pooling"]
    ):
        # GRID SEARCH PART
        hyperparams = load_hyperparams("resnet_params.json")
        trainer = initialize_model(model(), 
                                save_path=f"./models/bert_multitask_model{prefix}", 
                                results_path=f"./results/resnet/training_results_resnet{prefix}.csv",
                                data_type="image")

        best_params, best_val_acc = trainer.train(hyperparams=hyperparams)

        # TRAIN PART
        trainer = initialize_model(model(), 
                                save_path=f"./models/bert_multitask_model{prefix}_final", 
                                results_path=f"./results/resnet/training_results_resnet{prefix}_final.csv",
                                data_type="image")
        trainer.train(hyperparams=[best_params])