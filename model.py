import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base_llama import LlamaForCausalLMWithLora
from modules.base_qwen import Qwen2ForCausalLMWithLora
from transformers import AutoTokenizer
import math
from modules.lora import LoRAConfig
from utils import clean_text


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

        # Base model with incremental lora
        if 'Llama' in base_model_name:
            self.base_model = LlamaForCausalLMWithLora.from_pretrained(
                base_model_name,
                self.lora_config,
                attn_implementation='eager',  # To obtain attention weights
                torch_dtype=torch.bfloat16,
            )
        elif 'Qwen' in base_model_name:
            self.base_model = Qwen2ForCausalLMWithLora.from_pretrained(
                base_model_name,
                self.lora_config,
                attn_implementation='eager',  # To obtain attention weights
                torch_dtype=torch.bfloat16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.model_dim = self.base_model.config.hidden_size
        
        # lora attn key
        self.frozen_lora_keys = nn.ParameterList()
        self.trainable_lora_key = nn.Parameter(torch.empty(0, self.model_dim))

        # Input transform to lora attn query
        self.lora_query = nn.Sequential(
            nn.Linear(self.model_dim, lora_query_hidden_dim, bias=False),
            nn.Linear(lora_query_hidden_dim, self.model_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(self.model_dim)
        )
    
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
        # set device
        self.to(self.base_model.device)
        print("==== New task reset complete ====")

    def get_lora_attention(self, text_ids):
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

        return attn_weights  

    def forward(
        self, 
        text_ids=None, 
        input_ids=None, 
        attention_mask=None, 
        labels=None, 
        n_module_to_use=None
    ):
        # get lora attention
        lora_attn_weights = self.get_lora_attention(text_ids)

        # forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            lora_attn_weights=lora_attn_weights,
            output_attentions=True,
            n_module_to_use=n_module_to_use
        )

        return outputs

    def compute_distill_loss(self, head_importance, student_attns, teacher_attns):
        loss = 0.0
        for layer_idx, (s_attn, t_attn) in enumerate(zip(student_attns, teacher_attns)):
            _, num_heads, seq_len, _ = s_attn.shape
            # Get head importance of current layer
            layer_weight = head_importance.get(f"layer_{layer_idx}", 1.0).view(1, num_heads, 1, 1)
            
            loss += F.kl_div(
                torch.log(s_attn * layer_weight + 1e-8).view(-1, seq_len), 
                t_attn.detach().view(-1, seq_len), 
                reduction='batchmean',
                log_target=False,
            )
                
        return loss / len(student_attns)
        
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

        outputs = self.base_model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            lora_attn_weights=lora_attn_weights,
            max_new_tokens=256,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return clean_text(full_text)


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
