from classic_analysis.base import MultiTaskTrainer
from classic_analysis.bert_pipeline.model import BertLinear, BertMLP, BertDeepMLP
from utils.read_config import load_hyperparams

if __name__ == "__main__":
    model = BertMLP()

    hyperparams = load_hyperparams("bert_params.json")

    trainer = MultiTaskTrainer(
        model=model,
        csv_path="data/memotion_dataset_7k/labels.csv",
        data_type="text",
        save_path="./models/bert_multitask_model_mlp",
        results_path="./results/bert/training_results_bert_mlp.csv",
        use_mlflow=True,
    )

    best_params, best_val_acc = trainer.train(hyperparams=hyperparams)

    print(f"Best hyperparameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
