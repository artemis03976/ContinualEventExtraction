import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base_llama import LlamaForCausalLMWithLora
from transformers import AutoTokenizer
import math

from modules.lora import LoRAConfig


class ContinualEventExtractionModel(nn.Module):
    def __init__(
        self, 
        base_model_name='',
        lora_query_hidden_dim=1024,
        lora_r=32,
        lora_alpha=1,
        lora_dropout=0.0,
    ):
        super().__init__()

        self.lora_config = LoRAConfig(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # base model with incremental lora
        self.base_model = LlamaForCausalLMWithLora.from_pretrained(
            base_model_name,
            self.lora_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.model_dim = self.base_model.config.hidden_size
        
        # lora attn key
        self.frozen_lora_keys = nn.ParameterList()
        self.trainable_lora_key = nn.Parameter(torch.empty(0, self.model_dim))

        # input transform to lora attn query
        self.lora_query = nn.Sequential(
            nn.Linear(self.model_dim, lora_query_hidden_dim, bias=False),
            nn.Linear(lora_query_hidden_dim, self.model_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(self.model_dim)
        )

        # 添加注意力缓存
        self.hist_attn_cache = []  # 存储历史任务的注意力原型
        self.current_attn_proto = None  # 当前任务注意力原型
    
    def add_new_lora_key(self):
        # Freeze all the previous lora keys
        if len(self.trainable_lora_key) > 0:
            frozen_param = nn.Parameter(self.trainable_lora_key.data.clone().detach())
            frozen_param.requires_grad = False
            self.frozen_lora_keys.append(frozen_param)

        new_param = nn.Parameter(torch.empty(1, self.model_dim))
        nn.init.uniform_(new_param, -1, 1)
        self.trainable_lora_key = new_param

    def add_new_lora_layer(self):
        for layer in self.base_model.model.layers:
            layer.self_attn.add_new_lora()

    def reset_for_new_task(self):
        self.add_new_lora_key()
        print("==== New lora key initialize done ====")
        self.add_new_lora_layer()
        print("==== New lora layer initialize done ====")
        self.cache_attention_prototype()
        self.current_attn_proto = None
        print("==== Cache attention prototype done ===")
        # set device
        self.to(self.base_model.device)
        print("==== New task reset complete ====")
    
    def cache_attention_prototype(self):
        if self.current_attn_proto is not None:
            frozen_proto = self.current_attn_proto.detach().clone()
            self.hist_attn_cache.append(frozen_proto)

    def get_lora_attention(self, text_ids, record_proto=True):
        input_embeds = self.base_model.get_input_embeddings()(text_ids)

        # max-pool operation
        inputs_embeds = input_embeds.max(dim=1, keepdim=True).values
        lora_query = self.lora_query(inputs_embeds)
        
        # concat lora key
        all_lora_keys = []
        for param in self.frozen_lora_keys:
            all_lora_keys.append(param)
        all_lora_keys.append(self.trainable_lora_key)
        
        lora_key = torch.cat(all_lora_keys, dim=0).repeat(inputs_embeds.shape[0], 1, 1).to(self.trainable_lora_key.device)

        # get attn weights on lora layer
        attn_scores = torch.bmm(lora_key, lora_query.transpose(1, 2)) / math.sqrt(self.model_dim)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # 新增记录原型逻辑
        if record_proto and self.training:
            # 使用指数移动平均更新原型
            batch_proto = attn_weights.mean(dim=0)  # [n_keys, 1]

            if self.current_attn_proto is not None:
                if batch_proto.size(0) != self.current_attn_proto.size(0):
                    # 当键数量变化时重置原型
                    self.current_attn_proto = None

            if self.current_attn_proto is None:
                self.current_attn_proto = batch_proto
            else:
                self.current_attn_proto = 0.9 * self.current_attn_proto + 0.1 * batch_proto

        return attn_weights  

    def forward(self, text_ids, input_ids, attention_mask, labels):
        # get lora attention
        lora_attn_weights = self.get_lora_attention(text_ids)

        # forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            lora_attn_weights=lora_attn_weights
        )

        # 计算KL正则项
        kl_loss = self.compute_kl_penalty(lora_attn_weights)

        return outputs, kl_loss
    
    def compute_kl_penalty(self, current_attn):
        """
        计算新旧注意力分布的KL散度惩罚项
        current_attn: 当前batch的注意力权重 [B, n_keys, 1]
        """
        kl_loss = 0.0
        batch_size = current_attn.size(0)

        for hist_attn in self.hist_attn_cache:
            # 对齐维度：历史原型[n_keys,1] -> [1,n_keys,1]
            hist = hist_attn.unsqueeze(0).to(current_attn.device)
            hist_n_keys = hist.size(1)
            current_n_keys = current_attn.size(1)

            padding = torch.zeros(
                (1, current_n_keys - hist_n_keys, 1),
                device=current_attn.device
            )
            aligned_hist = torch.cat([hist, padding], dim=1)
            
            # 计算分布相似性（反向KL散度更稳定）
            kl_div = F.kl_div(
                input=current_attn.log(), 
                target=aligned_hist,
                reduction='batchmean',
                log_target=False
            )
            kl_loss += kl_div
            
        return kl_loss / len(self.hist_attn_cache) if self.hist_attn_cache else 0.0
        
    def save(self, path):
        lora_layer_name = ['lora_q_weights', 'lora_v_weights']
        lora_state_dict = {
            k: v for k, v in self.base_model.state_dict().items()
            if any(keyword in k for keyword in lora_layer_name)
        }

        lora_query_state_dict = self.lora_query.state_dict()

        frozen_lora_keys = [param.detach().clone() for param in self.frozen_lora_keys]
        # concat lora key list
        if len(self.trainable_lora_key) > 0:
            frozen_param = nn.Parameter(self.trainable_lora_key.clone().detach())
            frozen_lora_keys.append(frozen_param)
            temp_param_list = nn.ParameterList([
                nn.Parameter(p, requires_grad=False) for p in frozen_lora_keys
            ])
        
        lora_key_state_dict = temp_param_list.state_dict()

        combined_state_dict = {}
        # 添加 base_model 的 LoRA 权重
        for k, v in lora_state_dict.items():
            combined_state_dict[f'base_model.{k}'] = v

        # 添加 lora_query 的权重
        for k, v in lora_query_state_dict.items():
            combined_state_dict[f'lora_query.{k}'] = v

        # 添加 frozen_lora_keys 的权重
        for k, v in lora_key_state_dict.items():
            combined_state_dict[f'frozen_lora_keys.key_{k}'] = v

        torch.save(combined_state_dict, path)

    def load(self, path):
        combined_state_dict = torch.load(path)

        # 恢复 base_model 中的 LoRA 层
        base_model_lora_state = {k.replace('base_model.', ''): v for k, v in combined_state_dict.items() if k.startswith('base_model.')}
        self.base_model.load_state_dict(base_model_lora_state, strict=False)

        # 恢复 lora_query 层
        lora_query_state = {k.replace('lora_query.', ''): v for k, v in combined_state_dict.items() if k.startswith('lora_query.')}
        self.lora_query.load_state_dict(lora_query_state, strict=False)

        # 恢复 frozen_lora_keys
        self.frozen_lora_keys = nn.ParameterList()
        self.trainable_lora_key = nn.Parameter(torch.empty(0, self.model_dim))
        idx = 0
        while True:
            key = f'frozen_lora_keys.key_{idx}'
            if key in combined_state_dict:
                param = nn.Parameter(combined_state_dict[key])
                param.requires_grad = False
                self.frozen_lora_keys.append(param)
                idx += 1
            else:
                break

    def generate(self, text_ids, input_ids, attention_mask):
         # get lora attention
        lora_attn_weights = self.get_lora_attention(text_ids)

        return self.base_model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            lora_attn_weights=lora_attn_weights,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='maven')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    model = ContinualEventExtractionModel(
        base_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ).to("cuda")
    model.reset_for_new_task()

    # print(model)
    # print(model.frozen_lora_keys)
    # print(model.trainable_lora_key)

    from data import get_dataloader

    data_loader = get_dataloader(args, tokenizer=model.tokenizer, phase='trigger', split='train')

    with autocast(dtype=torch.bfloat16):
        for batch in data_loader:
            text_ids = batch['text_ids'].to("cuda")
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(text_ids, input_ids, attention_mask, labels)
            print(outputs.loss)
            break
