from classic_analysis.base import initialize_model
from classic_analysis.resnet_pipeline.model import (
    ResNetAdaptivePooling,
    ResNetAttention,
    ResNetLinear,
)


if __name__ == "__main__":

    for model, prefix in zip(
        [ResNetAttention, ResNetAdaptivePooling, ResNetLinear],
        ["_attention", "_pooling", "_linear"],
    ):
        trainer = initialize_model(model=model(), 
                                   save_path=f"./models/resnet_multitask_model{prefix}_final",
                                   test=True,
                                   data_type="image")
        trainer.test(f"./results/test/resnet_multitask_model{prefix}_final")
