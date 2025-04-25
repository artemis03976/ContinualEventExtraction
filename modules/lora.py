import torch
import torch.nn as nn
import math


class LoRAConfig(object):
    def __init__(self, lora_r, lora_alpha, lora_dropout):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_config: LoRAConfig,
    ):
        super(LoRALayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_config.lora_r
        self.lora_alpha = lora_config.lora_alpha
        self.lora_dropout = lora_config.lora_dropout

        self.lora_A = nn.Parameter(torch.zeros((self.lora_r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.lora_r)))

        self.scaling = self.lora_alpha / self.lora_r
        # Optional dropout
        if self.lora_dropout > 0.:
            self.dropout = nn.Dropout(p=self.lora_dropout)
        else:
            self.dropout = lambda x: x
        # Mark the weight as unmerged
        
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor):
        result = (self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)
