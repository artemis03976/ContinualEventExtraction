import torch
import json
import argparse
import os
from argparse import Namespace
from accelerate import Accelerator
from utils import set_seeds, compare_json
from model import ContinualEventExtractionModel
from data import get_dataloader
from tqdm import tqdm
import sys


def inference(args, model, accelerator):
    with open(os.path.join(args.checkpoint_path, 'event_type_groups.json'), 'r') as f:
        event_type_groups = json.load(f)
    test_dataloaders = get_dataloader(args, event_type_groups, tokenizer=model.tokenizer, phase='trigger', split='test')

    for task_id, test_dataloader in enumerate(test_dataloaders):
        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        acc = 0.0
        n_samples = 0
        n_invalid = 0

        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader):
                text_ids, input_ids, attention_mask, labels = batch['text_ids'], batch['input_ids'], batch['attention_mask'], batch['labels']

                # get input_ids without answer
                mask_lens = (labels == -100).sum(dim=1)
                input_ids_without_labels = input_ids[:, :mask_lens]
                attention_mask = attention_mask[:, :mask_lens]

                with accelerator.autocast():
                    outputs = model.generate(text_ids, input_ids_without_labels, attention_mask)

                labels = [sample[sample != -100.0] for sample in labels]
                labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

                results = compare_json(outputs, labels)

                for result in results:
                    if result != -1:
                        acc += result
                    else:
                        n_invalid += 1

                n_samples += len(text_ids)
        
        acc /= n_samples
        print(f'Accuracy: {acc}')
        print(f'Invalid: {n_invalid}')
    

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
    )
    for i in range(args.n_tasks):
        model.reset_for_new_task()
    model.load(os.path.join(args.checkpoint_path, f'ckpt_best_loss_task_{args.n_tasks}.pth'))

    torch.cuda.empty_cache()

    inference(args, model, accelerator)


if __name__ == "__main__":
    main()
