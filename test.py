import torch
import json
import argparse
import os
from argparse import Namespace
from accelerate import Accelerator
from utils import set_seeds, get_logger
from model import ContinualEventExtractionModel
from data import get_dataloader
from tqdm import tqdm


def inference(args, model, accelerator, logger):
    test_dataloader = get_dataloader(args, tokenizer=model.tokenizer, phase='trigger', split='test')

    model, optimizer, test_dataloader = accelerator.prepare(model, optimizer, test_dataloader)

    with torch.no_grad():
        model.eval()
        dev_loss = 0.0
        for batch in tqdm(test_dataloader):
            text_ids, input_ids, attention_mask = batch['text_ids'], batch['input_ids'], batch['attention_mask']
        
            outputs = model.generate(text_ids, input_ids, attention_mask)
            dev_loss += outputs.loss.item()


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
        mixed_precision='no', 
    )

    model = ContinualEventExtractionModel(
        base_model_name=args.base_model_name,
        lora_query_hidden_dim=args.lora_query_hidden_dim,
        attn_temperature=args.attn_temperature,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model.load(os.path.join(args.checkpoint_path, 'ckpt_best_loss.pth'))

    torch.cuda.empty_cache()

    logger = get_logger(args)

    inference(args, model, accelerator, logger)


if __name__ == "__main__":
    main()
