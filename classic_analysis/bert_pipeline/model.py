import torch.nn as nn
from transformers import AutoModel
import torch

from classic_analysis.base import MultiTaskModel


class BertLinear(MultiTaskModel):
    def __init__(self, model_name="bert-base-uncased", tasks=None, dropout=0.1):
        super().__init__(tasks)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, 2) for task in self.tasks}
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return {task: self.heads[task](pooled) for task in self.tasks}


class BertMLP(MultiTaskModel):
    def __init__(
        self, model_name="bert-base-uncased", tasks=None, dropout=0.2, hidden_dim=256
    ):
        super().__init__(tasks)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return {task: self.heads[task](pooled) for task in self.tasks}


class BertDeepMLP(MultiTaskModel):
    def __init__(
        self, model_name="bert-base-uncased", tasks=None, dropout=0.2, hidden_dim=256
    ):
        super().__init__(tasks)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token
        cls_token = outputs.last_hidden_state[:, 0]

        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        )
        sum_hidden = (last_hidden * attention_mask_expanded).sum(1)
        sum_mask = attention_mask_expanded.sum(1)
        mean_pooled = sum_hidden / sum_mask

        combined = torch.cat([cls_token, mean_pooled], dim=1)
        projected = self.feature_projection(combined)

        return {task: self.heads[task](projected) for task in self.tasks}
