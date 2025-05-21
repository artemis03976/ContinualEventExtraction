import torch
import json
import argparse
import os
from argparse import Namespace
from accelerate import Accelerator
from utils import set_seeds, compute_event_f1, compute_acc
from model import ContinualEventExtractionModel
from data import get_dataloader
from tqdm import tqdm


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

    with open(os.path.join(args.checkpoint_path, 'predictions.jsonl'), 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    

def evaluate(args):
    total_results = {}
    with open(os.path.join(args.checkpoint_path, 'predictions.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            task_key = f"task_{item['task_id']}"
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
    
    task_acc = 0.0
    for task_id, results in total_results.items():
        task_f1_score = compute_event_f1(results)
        print(f"Task {task_id} F1: {task_f1_score['f1']:.4f}")

        task_acc += compute_acc(results)
    
    print(f"Acc: {task_acc / len(total_results):.4f}")
        

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

    model = ContinualEventExtractionModel(
        base_model_name=args.base_model_name,
        lora_query_hidden_dim=args.lora_query_hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        disable_shared_attn=args.disable_shared_attn,
    )
    for i in range(args.n_tasks):
        model.reset_for_new_task()
    # model.reset_for_new_task()
    model.load(os.path.join(args.checkpoint_path, f'ckpt_best_loss_task_{args.n_tasks}.pth'))

    torch.cuda.empty_cache()

    with open(os.path.join(args.checkpoint_path, 'event_type_groups.json'), 'r') as f:
        event_type_groups = json.load(f)
    test_dataloaders = get_dataloader(args, event_type_groups, tokenizer=model.tokenizer, phase='trigger', split='test')

    inference(args, model, accelerator, test_dataloaders)
    evaluate(args)


if __name__ == "__main__":
    main()
