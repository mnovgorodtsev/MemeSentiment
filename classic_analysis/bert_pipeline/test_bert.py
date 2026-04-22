from classic_analysis.base import MultiTaskTrainer
from classic_analysis.bert_pipeline.model import BertLinear, BertMLP, BertDeepMLP

if __name__ == "__main__":

    for model, prefix in zip(
        [BertLinear, BertMLP, BertDeepMLP], ["_linear", "_mlp", "_mlp_deep"]
    ):
        trainer = MultiTaskTrainer(
            model=model(),
            csv_path="data/memotion_dataset_7k/labels.csv",
            data_type="text",
            save_path=f"./models/bert_multitask_model{prefix}",
            test=True,
        )

        trainer.test()
