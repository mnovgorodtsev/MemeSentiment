from classic_analysis.resnet_bert_fusion.late_fusion.model import LateFusionTrainer

if __name__ == "__main__":

    trainer = LateFusionTrainer(
        csv_path="data/memotion_dataset_7k/labels.csv",
        images_dir="data/memotion_dataset_7k/images",
        resnet_path="./models/resnet_multitask_model/model_weights.pt",
        bert_path="./models/bert_multitask_model/model_weights.pt",
    )

    trainer.test()
