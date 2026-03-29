import torch.nn as nn
from transformers import AutoModel

from classic_analysis.base import MultiTaskModel


class BertMultiTaskModel(MultiTaskModel):
    def __init__(self, model_name="bert-base-uncased", tasks=None):
        super().__init__(tasks)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, 2) for task in self.tasks}
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return {task: self.heads[task](pooled) for task in self.tasks}
