import torch
from data import collate_fn
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import broadcast
from functools import partial


class HardSampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class HardSampleBuffer:
    def __init__(self, tokenizer=None, max_buffer_size=200, ratio=0.1, mode='distribution'):
        self.buffer = {}  # { task_id: task buffer } 
        self.tmp_batch_buffer = []
        self.tmp_loss_buffer = []

        self.tokenizer = tokenizer

        self.max_buffer_size = max_buffer_size
        self.ratio = ratio
        self.mode = mode

    def clear_tmp(self):
        self.tmp_batch_buffer = []
        self.tmp_loss_buffer = []

    def update(self, task_id, accelerator):
        if not self.tmp_batch_buffer:
            return
    
        # Step 1: Collect all local loss information
        local_losses = torch.tensor(self.tmp_loss_buffer, device=accelerator.device)
        gathered_losses = accelerator.gather(local_losses)

        # Step 2: Process loss threshold on main process
        if accelerator.is_main_process:
            gathered_losses = gathered_losses.view(-1)
            threshold = gathered_losses.mean() + 1.3 * gathered_losses.std()
        else:
            threshold = torch.tensor(0.0, device=accelerator.device)

        # Step 3: Broadcast threshold to all processes
        threshold = broadcast(threshold)

        # Step 4: Filter samples based on threshold
        mask = local_losses >= threshold
        if not mask.any() or self.mode == 'topk':
            k = max(1, int(self.ratio * len(local_losses)))
            _, indices = torch.topk(local_losses, k)
            mask = torch.zeros_like(local_losses).bool()
            mask[indices] = True
        selected_samples = [self.tmp_batch_buffer[i] for i in mask.nonzero().flatten().tolist()]
 
        if len(selected_samples) > self.max_buffer_size:
            selected_samples = selected_samples[:self.max_buffer_size]
        self.buffer[task_id] = selected_samples
    
        self.clear_tmp()
    
    def add(self, batch, losses):
        for i in range(len(losses)):
            sample = {
                'text_ids': batch['text_ids'][i].detach(),
                'input_ids': batch['input_ids'][i].detach(),
                'attention_mask': batch['attention_mask'][i].detach(),
                'labels': batch['labels'][i].detach(),
            }
            self.tmp_batch_buffer.append(sample)
            self.tmp_loss_buffer.append(losses[i].detach())

    def collect(self, device, batch_size):
        all_task_buffer = []
     
        for task_buffer in self.buffer.values():
            for item in task_buffer:
                sample = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in item.items()
                }
                all_task_buffer.append(sample)

        dataset = HardSampleDataset(all_task_buffer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer),
        )
    
    def is_tmp_empty(self):
        return len(self.tmp_batch_buffer) == 0 and len(self.tmp_loss_buffer) == 0
