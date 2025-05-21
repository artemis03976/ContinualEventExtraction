import random
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
import logging
from torch.nn.parallel import DistributedDataParallel
import json


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的 handler，避免重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if args.log_path is not None:
        os.makedirs(args.log_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(args.log_path, f'train_log_{timestamp}.log')
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def distribution_state_manager(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    else:
        return model


def compute_sample_losses(logits, labels):

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    losses = []

    for i in range(logits.shape[0]):
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        losses.append(loss_fct(shift_logits[i], shift_labels[i])) 
    
    return losses


def get_grad_norm(attn_grad):
    # attn_grad: (batch_size, num_heads, seq_len, seq_len)
    # 先按 batch 维度求平均，然后对每个 head 计算 L2 范数
    grad_mean = attn_grad.mean(dim=0)  # (num_heads, seq_len, seq_len)
    # 展开最后两个维度，计算每个 head 的 L2 范数
    num_heads = grad_mean.size(0)
    l2_norms = grad_mean.view(num_heads, -1).norm(p=2, dim=1)  # (num_heads,)
    return l2_norms


def clean_text(texts):
    cleaned_text = []
    for text in texts:
        marker = "Results:"
        start_idx = text.find(marker)
        if start_idx != -1:
            text = text[start_idx + len(marker):].strip()

        marker = "`\n</think>"
        start_idx = text.find(marker)
        if start_idx != -1:
            text = text[:start_idx].strip()

        cleaned_text.append(text)

    return cleaned_text


def get_mask_len(prompt, tokenizer):
    token_id = tokenizer.encode(":", add_special_tokens=False)[0]

    reverse_index = torch.where(prompt.flip(0) == token_id)[0]

    if len(reverse_index) > 0:
        idx = prompt.size(0) - 1 - reverse_index[0].item()
        result = prompt[:idx + 1]

    return len(result)


def compute_acc(results):
    acc = 0.0

    for (pred_events, gold_events) in results:
        if pred_events == json.loads('[]'):
            pred_events = json.loads('[{"trigger_word": "None", "event_type": "None", "span": "None"}]')
        try:
            pred_set = set((e['event_type'], e['trigger_word'], str(e['span'])) for e in pred_events)
        except:
            pred_set = set()
        gold_set = set((e['event_type'], e['trigger_word'], str(e['span'])) for e in gold_events)

        acc += int(pred_set == gold_set)

    return acc / len(results) * 100

    
def compute_event_f1(results):
    TP, FP, FN = 0, 0, 0

    for (pred_events, gold_events) in results:
        if pred_events == json.loads('[]'):
            pred_events = json.loads('[{"trigger_word": "None", "event_type": "None", "span": "None"}]')
        try:
            pred_set = set((e['event_type'], e['trigger_word'], str(e['span'])) for e in pred_events)
        except:
            pred_set = set()
        gold_set = set((e['event_type'], e['trigger_word'], str(e['span'])) for e in gold_events)

        TP += len(pred_set & gold_set)
        FP += len(pred_set - gold_set)
        FN += len(gold_set - pred_set)

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }