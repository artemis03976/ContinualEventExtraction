import torch
import torch.optim as optim
import argparse
import os
import json
from utils import set_seeds, get_logger, distribution_state_manager
from data import get_dataloader, get_event_types, split_tasks_by_event_types
from tqdm import tqdm
from model import ContinualEventExtractionModel
from accelerate import Accelerator


def train_single_task(args, model, train_dataloader, dev_dataloaders, accelerator, logger, task_id):
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model_interface = distribution_state_manager(model)

    total_loss_history = {
        'train_loss': [],
        'dev_loss': [],
    }

    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            logger.info("=========================================")
            logger.info(f"==== Epoch: {epoch} / {args.epochs} ====")

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # new task sample
                outputs = model(**batch)
                accelerator.backward(outputs.loss)

                optimizer.step()

                train_loss += outputs.loss.item()

        train_loss /= len(train_dataloader)
        total_loss_history['train_loss'].append(train_loss)

        # evaluation
        with torch.no_grad():
            model.eval()
            total_dev_loss = 0.0
            for dev_dataloader in dev_dataloaders:
                dev_loss = 0.0
                dev_dataloader = accelerator.prepare(dev_dataloader)

                for batch in tqdm(dev_dataloader):
                    outputs = model(**batch)
                    dev_loss += outputs.loss.item()

                total_dev_loss += dev_loss / len(dev_dataloader)
        total_dev_loss /= len(dev_dataloaders)
        total_loss_history['dev_loss'].append(total_dev_loss)

        best_loss = min(total_loss_history['dev_loss'])
        best_loss_epoch = total_loss_history['dev_loss'].index(best_loss) + 1
        if accelerator.is_main_process:
            logger.info(f"==== Total Train Loss for Epoch {epoch}: {train_loss} ====")
            logger.info(f"==== Total Dev Loss for Epoch {epoch}: {total_dev_loss} ====")
            logger.info(f"==== Best Dev Loss: {best_loss} at Epoch {best_loss_epoch} ====")

        # save best epoch
        if epoch == best_loss_epoch:
            if accelerator.is_main_process:
                ckpt_path = os.path.join(args.output_path, f"ckpt_best_loss_task_{task_id}.pth")
                model_interface.save(ckpt_path)
                logger.info(f"==== Save best loss ckpt to {ckpt_path} ====")
        
        accelerator.wait_for_everyone()

    # save in the end
    if accelerator.is_main_process:
        ckpt_path = os.path.join(args.output_path, f"ckpt_final_task_{task_id}.pth")
        model_interface.save(ckpt_path)
        logger.info(f"==== Save final ckpt to {ckpt_path} ====")

    accelerator.wait_for_everyone()


def train(args, model, accelerator, logger):
    # Get all event types
    event_types = get_event_types(args, force_rebuild=True)
    event_type_groups = split_tasks_by_event_types(args, event_types=event_types)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_path, 'event_type_groups.json'), 'w') as f:
            json.dump(event_type_groups, f, indent=4)

    train_dataloaders = get_dataloader(args, event_type_groups, tokenizer=model.tokenizer, phase='trigger', split='train')
    dev_dataloaders = get_dataloader(args, event_type_groups, tokenizer=model.tokenizer, phase='trigger', split='dev')

    for task_id, (train_dataloader, dev_dataloader) in enumerate(zip(train_dataloaders, dev_dataloaders)):
        if accelerator.is_main_process:
            logger.info(f"==== Training Task {task_id + 1} ====")
            logger.info(f"==== Initialize model for task {task_id + 1} ====")
        
        # Load best checkpoint from last task before training the new one
        if task_id > 0:
            previous_ckpt_path = os.path.join(args.output_path, f"ckpt_best_loss_task_{task_id}.pth")
            model.load(previous_ckpt_path)
            if accelerator.is_main_process:
                logger.info(f"Loaded best checkpoint from task {task_id} at {previous_ckpt_path}")

        model.reset_for_new_task()
        train_single_task(args, model, train_dataloader, dev_dataloaders[:task_id + 1], accelerator, logger, task_id + 1)

        if accelerator.is_main_process:
            logger.info(f"==== Task {task_id + 1} Training Done ====")

        torch.cuda.empty_cache()
        


def get_args():
    parser = argparse.ArgumentParser()

    # base model settings
    parser.add_argument(
        '--base_model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', 
        help='base model for generating answers'
    )
    parser.add_argument(
        '--lora_query_hidden_dim', type=int, default=128, 
        help=''
    )
    parser.add_argument(
        '--lora_r', type=int, default=16, 
        help=''
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=16, 
        help=''
    )
    parser.add_argument(
        '--lora_dropout', type=float, default=0.1,
        help=''
    )

    # data settings
    parser.add_argument(
        '--n_tasks', type=int, default=5, 
        help='number of tasks for continual event extraction'
    )
    parser.add_argument(
        '--dataset', type=str, default='maven',
        help='dataset type'
    )
    parser.add_argument(
        '--data_root', type=str, default='./data/MAVEN',
        help='dataset path'
    )
    parser.add_argument(
        '--strategy', type=str, default='sequential',
        help='stratrgy for spliting dataset '
    )

    # training settings
    parser.add_argument(
        '--seed', type=int, default=42, 
        help='random seed'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help='Learning rate of network.'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, 
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Training batch size. Set to n_samples by default.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=2,
        help='gradient accumulation steps during training'
    )
    parser.add_argument(
        '--disable_shared_attn', action='store_true', default=False,
        help='是否启用共享注意力'
    )

    # save settings
    parser.add_argument(
        '--log_path', type=str, default=None,
        help='path to log th results'
    )
    parser.add_argument(
        '--output_path', type=str, default='./checkpoints',
        help='path to save the model'
    )

    args = parser.parse_args()

    # print and save the args
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return args


def main():
    args = get_args()
    set_seeds(args.seed)

    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    model = ContinualEventExtractionModel(
        base_model_name=args.base_model_name,
        lora_query_hidden_dim=args.lora_query_hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        disable_shared_attn=args.disable_shared_attn,
    )

    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger = get_logger(args)
    else:
        logger = None

    train(args, model, accelerator, logger)


if __name__ == '__main__':
    main()
