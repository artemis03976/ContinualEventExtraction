import torch
import json
import argparse
import os
from argparse import Namespace
from accelerate import Accelerator
from utils import set_seeds, compute_event_f1
from model import ContinualEventExtractionModel
from data import get_dataloader
from tqdm import tqdm
import numpy as np


class PerformanceLogger:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.R = np.full((n_tasks+1, n_tasks+1), -1.0)  # 索引从1开始
        
    def update(self, args, task_id):
        """记录当前模型在所有已训练任务上的性能"""
        total_results = {}
        with open(os.path.join(args.checkpoint_path, f'predictions_step_{task_id}.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                task_key = item['task_id']
                if task_key not in total_results:
                    total_results[task_key] = []

                try:
                    pred = json.loads(item['prediction'])
                except json.JSONDecodeError:
                    pred = json.loads('[]')

                total_results[task_key].append((
                    pred, 
                    json.loads(item['label'])
                ))
        for eval_task_id, results in total_results.items():
            task_f1_score = compute_event_f1(results)['f1']
            self.R[task_id][int(eval_task_id)] = task_f1_score
                
    def calculate_bwt(self):
        """计算反向迁移分数"""
        bwt_scores = []
        for j in range(1, self.n_tasks):
            final_perf = self.R[self.n_tasks][j]
            initial_perf = self.R[j][j]
            if final_perf == -1 or initial_perf == -1:
                continue
            bwt_scores.append(final_perf - initial_perf)
        return np.mean(bwt_scores) if bwt_scores else 0


def inference(args, model, accelerator, test_dataloaders):
    results = []
    for task_id, test_dataloader in enumerate(test_dataloaders):
        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader):
                text_ids, input_ids, attention_mask, labels = batch['text_ids'], batch['input_ids'], batch['attention_mask'], batch['labels']

                with accelerator.autocast():
                    outputs = model.generate(text_ids, input_ids, attention_mask)

                for i in range(len(outputs)):
                    results.append({
                        'task_id': task_id + 1,
                        'label': labels[i],
                        'prediction': outputs[i]
                    })
    
    return results
        

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_path', type=str, default='./checkpoints',
        help='path to save the model'
    )

    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_path, "config.json"), "r") as f:
        args_dict = json.load(f)
    base_args = Namespace(**args_dict)

    merged = vars(base_args).copy()
    merged.update(vars(args))

    return Namespace(**merged)


def main():
    args = get_args()

    set_seeds(args.seed)

    accelerator = Accelerator(
        mixed_precision='bf16', 
    )

    perf_logger = PerformanceLogger(n_tasks=args.n_tasks)

    model = ContinualEventExtractionModel(
        base_model_name=args.base_model_name,
        lora_query_hidden_dim=args.lora_query_hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        disable_shared_attn=args.disable_shared_attn,
    )

    with open(os.path.join(args.checkpoint_path, 'event_type_groups.json'), 'r') as f:
        event_type_groups = json.load(f)
        test_dataloaders = get_dataloader(args, event_type_groups, tokenizer=model.tokenizer, phase='trigger', split='test')

    # model.reset_for_new_task()
    for i in range(args.n_tasks):
        print(f"==== Evaluating checkpoint for Task {i + 1} ====")
        model.reset_for_new_task()
        if os.path.exists(os.path.join(args.checkpoint_path, f'predictions_step_{i + 1}.jsonl')):
            perf_logger.update(args, i + 1)
            continue
        model.load(os.path.join(args.checkpoint_path, f'ckpt_best_loss_task_{i + 1}.pth'))

        results = inference(args, model, accelerator, test_dataloaders[:i + 1])

        with open(os.path.join(args.checkpoint_path, f'predictions_step_{i + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        perf_logger.update(args, i + 1)
        torch.cuda.empty_cache()

    bwt_score = perf_logger.calculate_bwt()
    np.save(os.path.join(args.checkpoint_path, 'performance_matrix.npy'), perf_logger.R)
    print(f"\nBackward Transfer Score: {bwt_score:.4f}")


if __name__ == "__main__":
    main()
