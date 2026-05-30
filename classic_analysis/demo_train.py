from classic_analysis.base import initialize_model
from classic_analysis.bert_pipeline.model import BertLinear
from classic_analysis.resnet_pipeline.model import ResNetLinear
from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer
from classic_analysis.resnet_bert_fusion.late_fusion.model import LateFusionTrainer
from utils.read_config import Config, load_hyperparams

DEMO_PARAMS = [{"batch_size": 8, "epochs": 1, "learning_rate": 2e-5, "lr_finetune": 1e-4}]
DEMO_LATE_PARAMS = [{"w_image": 0.5, "w_text": 0.5}]


if __name__ == "__main__":
    config = Config()

    CSV_PATH = config.memotion_dataset_path
    IMAGES_DIR = config.images_base_path
    RESULTS_DIR = config.results_classic_dir

    BERT_DEMO_PATH = f"./models/demo/bert_final"
    RESNET_DEMO_PATH = f"./models/demo/resnet_final"

    print("\n=== [1/4] BERT ===")
    trainer = initialize_model(BertLinear(),
                               save_path=BERT_DEMO_PATH,
                               csv_path=CSV_PATH,
                               results_path=f"{RESULTS_DIR}/bert.csv")
    trainer.train(hyperparams=DEMO_PARAMS)

    print("\n=== [2/4] ResNet ===")
    trainer = initialize_model(ResNetLinear(),
                               save_path=RESNET_DEMO_PATH,
                               csv_path=CSV_PATH,
                               images_dir=IMAGES_DIR,
                               results_path=f"{RESULTS_DIR}/resnet.csv",
                               data_type="image")
    trainer.train(hyperparams=DEMO_PARAMS)

    print("\n=== [3/4] Early Fusion ===")
    trainer = EarlyFusionTrainer(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        save_path=f"{RESULTS_DIR}/early_fusion.pt",
        results_path=f"{RESULTS_DIR}/early_fusion.csv",
        use_mlflow=False,
    )
    trainer.train(hyperparams=DEMO_PARAMS)

    print("\n=== [4/4] Late Fusion ===")
    trainer = LateFusionTrainer(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        resnet_path=f"{RESNET_DEMO_PATH}/model_weights.pt",
        bert_path=f"{BERT_DEMO_PATH}/model_weights.pt",
        results_path=f"{RESULTS_DIR}/late_fusion.csv",
        use_mlflow=False,
        text_model=BertLinear(),
        image_model=ResNetLinear(),
    )
    best_params, best_acc = trainer.train(hyperparams=DEMO_LATE_PARAMS)
    print(f"Late fusion best params: {best_params}, acc: {best_acc}")