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
                                   test=True,
                                   data_type="image")
        trainer.test()
