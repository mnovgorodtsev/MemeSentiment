from classic_analysis.base import initialize_model
from classic_analysis.bert_pipeline.model import BertLinear, BertMLP, BertDeepMLP
from utils.read_config import load_hyperparams


if __name__ == "__main__":
    for model, prefix in zip(
        [BertLinear, BertDeepMLP, BertMLP], ["_linear", "_mlp_deep", "_mlp"]
    ):
        # GRID SEARCH PART
        hyperparams = load_hyperparams("bert_params.json")
        trainer = initialize_model(model(), 
                                save_path=f"./models/bert_multitask_model{prefix}", 
                                results_path=f"./results/bert/training_results_bert{prefix}.csv")

        best_params, best_val_acc = trainer.train(hyperparams=hyperparams)

        # TRAIN PART
        trainer = initialize_model(model(), 
                                save_path=f"./models/bert_multitask_model{prefix}_final", 
                                results_path=f"./results/bert/training_results_bert{prefix}_final.csv")
        trainer.train(hyperparams=[best_params])