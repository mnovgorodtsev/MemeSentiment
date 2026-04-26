from classic_analysis.base import initialize_model
from classic_analysis.bert_pipeline.model import BertLinear, BertMLP, BertDeepMLP


if __name__ == "__main__":
    for model, prefix in zip(
        [BertLinear, BertMLP, BertDeepMLP], ["_linear", "_mlp", "_mlp_deep"]
    ):
        trainer = initialize_model(model=model(), 
                                   save_path=f"./models/bert_multitask_model{prefix}_final",
                                   test=True)
        trainer.test(f"./results/test/bert_multitask_model{prefix}_final")
