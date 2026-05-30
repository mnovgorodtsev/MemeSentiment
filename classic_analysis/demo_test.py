from classic_analysis.base import initialize_model
from classic_analysis.bert_pipeline.model import BertLinear
from classic_analysis.resnet_pipeline.model import ResNetLinear
from classic_analysis.resnet_bert_fusion.early_fusion.model import EarlyFusionTrainer
from classic_analysis.resnet_bert_fusion.late_fusion.model import LateFusionTrainer
from utils.read_config import Config

if __name__ == "__main__":
    config = Config()

    RESULTS_DIR = config.results_classic_dir
    BERT_DEMO_PATH = f"./models/demo/bert_final"
    RESNET_DEMO_PATH = f"./models/demo/resnet_final"

    print("\n=== [1/4] BERT ===")
    trainer = initialize_model(model=BertLinear(),
                               save_path=BERT_DEMO_PATH,
                               test=True)
    trainer.test(f"{RESULTS_DIR}/test/bert")

    print("\n=== [2/4] ResNet ===")
    trainer = initialize_model(model=ResNetLinear(),
                               save_path=RESNET_DEMO_PATH,
                               test=True,
                               data_type="image")
    trainer.test(f"{RESULTS_DIR}/test/resnet")

    print("\n=== [3/4] Early Fusion ===")
    trainer = EarlyFusionTrainer(
        save_path=f"{RESULTS_DIR}/early_fusion.pt",
        use_mlflow=False,
    )
    trainer.test(f"{RESULTS_DIR}/test/early_fusion")

    print("\n=== [4/4] Late Fusion ===")
    trainer = LateFusionTrainer(
        resnet_path=f"{RESNET_DEMO_PATH}/model_weights.pt",
        bert_path=f"{BERT_DEMO_PATH}/model_weights.pt",
        results_path=f"{RESULTS_DIR}/late_fusion.csv",
        use_mlflow=False,
        text_model=BertLinear(),
        image_model=ResNetLinear(),
    )
    trainer.test(f"{RESULTS_DIR}/test/late_fusion")