import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from prompt_template import base_model_prompt
from functools import partial
import random


def split_tasks_by_event_types(event_types, n_tasks, strategy='sequential'):
    """
    Split event types for task sequences
    Args:
        event_types: list of all event types
        n_tasks: number of tasks
        strategy: split strategy ('sequential'|'random'|'balanced'|'clustered')
    """
    if strategy == 'sequential':
        types_per_task = len(event_types) // n_tasks
        return [
            event_types[i * types_per_task : (i + 1) * types_per_task] 
            for i in range(n_tasks)
        ]

    elif strategy == 'random':
        shuffled = random.sample(event_types, len(event_types))
        return [shuffled[i::n_tasks] for i in range(n_tasks)]
    
    elif strategy == 'balanced':
        # 按数据量平衡划分（需预统计类型分布）
        # 此处需实现数据量统计逻辑
        raise NotImplementedError
    
    elif strategy == 'clustered':
        # 基于语义聚类划分（需类型嵌入表示）
        # 此处需实现聚类算法
        raise NotImplementedError
    
    else:
        raise ValueError(f"未知划分策略: {strategy}")


def get_event_types(args, force_rebuild=False):
    """
    Load event type from existing event_type.json, otherwise construct it from raw data
    """
    # 优先读取缓存文件
    cache_path = os.path.join(args.data_root, 'event_type.json')
    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 若需重建，直接从文件系统读取原始数据（不实例化完整Dataset）
    event_types = set()
    for split in ['train', 'valid', 'dev', 'test']:  # 遍历所有可能的分割文件
        file_path = os.path.join(args.data_root, f"{split}.jsonl")
        if not os.path.exists(file_path):
            continue
        
        if args.dataset == 'maven':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    for event in doc.get('events', []):
                        event_types.add(event['type'])
    
    # 保存并返回
    event_types = sorted(list(event_types))
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(event_types, f, ensure_ascii=False, indent=4)

    return event_types


class ContinualEventExtractionDataset(Dataset):
    def __init__(self, root='./data/', phase='trigger', split='train', tokenizer=None, event_types=None):
        self.root = root
        self.phase = phase
        self.split = split
        self.tokenizer = tokenizer
        self.event_types = event_types

        # load raw data
        self.raw_data = []
        file_path = os.path.join(self.root, "{}.jsonl".format(self.split))
        with open(file_path, "r", encoding="utf-8") as f:
            for data in f:
                docs = json.loads(data)
                self.raw_data.append(docs)
        
        # data for trigger classification
        self.trigger_data = None
        # data for argument classification
        self.argument_data = None

        # construct corresponding dataset
        if self.phase == 'trigger':
            self.construct_trigger_dataset()
        elif self.phase == 'argument':
            self.construct_argument_dataset()
        
        self.filter_data_by_event_type()

    def construct_trigger_dataset(self, force_rebuild=False):
        raise NotImplementedError

    def construct_argument_dataset(self, force_rebuild=False):
        raise NotImplementedError

    def filter_data_by_event_type(self):
        """
        Filter datas that do not contains current event types
        """
        if self.event_types is None:
            return
        
        # 处理trigger阶段数据
        if self.phase == 'trigger':
            filtered_data = []
            for item in self.trigger_data:
                events = json.loads(item['target_output'])
                # 检查是否存在当前任务允许的事件类型
                valid = all(event['event_type'] in self.event_types for event in events)
                if valid:
                    filtered_data.append(item)
            self.trigger_data = filtered_data
        
        # 处理argument阶段数据（类似逻辑）
        elif self.phase == 'argument':
            # ... 实现argument数据的过滤
            pass

    def __len__(self):
        if self.phase == 'trigger':
            return len(self.trigger_data)
        elif self.phase == 'argument':
            return len(self.argument_data)

    def build_prompt(self, input_text, target_output):
        messages = [
            {"role": "system", "content": base_model_prompt["system"]},
            {"role": "user", "content": base_model_prompt["user"].format(input_text=input_text)},
            {"role": "assistant", "content": base_model_prompt["assistant_prefix"]}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        full_prompt = prompt + target_output

        # 1. Construct input ids
        inputs = self.tokenizer(
            full_prompt,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = inputs["input_ids"][0]

        # 2. Construct labels
        prompt_only_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]
        labels = input_ids.clone()
        labels[:len(prompt_only_ids)] = -100 

        # 3. Construct text ids
        text_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        return text_ids, input_ids, inputs["attention_mask"], labels

    def __getitem__(self, idx):
        if self.phase == 'trigger':
            input_text, target_output = self.trigger_data[idx]['input_text'], self.trigger_data[idx]['target_output']
        elif self.phase == 'argument':
            input_text, target_output = self.argument_data[idx]['input_text'], self.argument_data[idx]['target_output']

        text_ids, input_ids, attention_mask, labels = self.build_prompt(input_text, target_output)
            
        return {
            'text_ids': text_ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels,
        }


class MavenDataset(ContinualEventExtractionDataset):
    def __init__(self, root='./data/MAVEN', phase='trigger', split='train', tokenizer=None, event_types=None):
        super().__init__(root, phase, split, tokenizer, event_types)

    def construct_trigger_dataset(self, force_rebuild=False):
        cache_path = os.path.join(self.root, f'{self.phase}_{self.split}.json')
        if not force_rebuild and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                self.trigger_data = json.load(f)
            return

        trigger_data = []
        for docs in self.raw_data:
            # get event text
            texts = [content['sentence'] for content in docs['content']]

            # get events and classify by sentence id
            all_events = {i: [] for i in range(len(texts))}
            for event in docs['events']:
                for mention in event['mention']:
                    sent_id = mention['sent_id']

                    # discard useless info
                    processed_event = {
                        'trigger_word': mention['trigger_word'],
                        'event_type': event['type'],
                        'span': mention['offset']
                    }
                    all_events[sent_id].append(processed_event)

            for i, text in enumerate(texts):
                trigger_data.append({
                    'input_text': text,
                    'target_output': json.dumps(all_events[i])
                })
        
        self.trigger_data = trigger_data
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(trigger_data, f, ensure_ascii=False, indent=4)


def collate_fn(batch, tokenizer):
    # 计算各序列的最大长度
    max_text_len = max(len(item["text_ids"]) for item in batch)
    max_input_len = max(len(item["input_ids"]) for item in batch)
    
    padded_batch = {
        "input_ids": [],
        "text_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for item in batch:
        # 填充 text_ids
        padding = [tokenizer.pad_token_id] * (max_text_len - len(item["text_ids"]))
        padded_seq = torch.cat([item["text_ids"], torch.tensor(padding, dtype=torch.long)])
        padded_batch['text_ids'].append(padded_seq)
    
        # 填充 input_ids
        padding = [tokenizer.pad_token_id] * (max_input_len - len(item["input_ids"]))
        padded_seq = torch.cat([item["input_ids"], torch.tensor(padding, dtype=torch.long)])
        padded_batch['input_ids'].append(padded_seq)
    
        # 填充 attention_mask
        padding = [0] * (max_input_len - len(item["input_ids"]))
        padded_seq = torch.cat([item["attention_mask"], torch.tensor(padding, dtype=torch.long)])
        padded_batch['attention_mask'].append(padded_seq)

        # 填充 labels
        padding = [-100] * (max_input_len - len(item["labels"]))
        padded_seq = torch.cat([item["labels"], torch.tensor(padding, dtype=torch.long)])
        padded_batch['labels'].append(padded_seq)

    return {
        "text_ids": torch.stack(padded_batch['text_ids']),
        "input_ids": torch.stack(padded_batch['input_ids']),
        "attention_mask": torch.stack(padded_batch['attention_mask']),
        "labels": torch.stack(padded_batch['labels']),
    }


def get_dataloader(args, event_type_groups, phase='trigger', split='train', tokenizer=None):
    task_dataloaders = []
    seen_event_types = [] 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 若模型无 pad_token，用 eos_token 代替

    for task_id, event_type_per_task in enumerate(event_type_groups):
        if split == 'train':
            if args.dataset == 'maven':
                sub_task_dataset = MavenDataset(
                    root=args.data_root,
                    phase=phase,
                    split=split,
                    tokenizer=tokenizer,
                    event_types=event_type_per_task
                )
                print(f'==== {len(sub_task_dataset)} {split} samples in Task {task_id} ====')

        elif split == 'dev':
            if args.dataset == 'maven':
                seen_event_types.extend(event_type_per_task)
                sub_task_dataset = MavenDataset(
                    root=args.data_root,
                    phase=phase,
                    split='valid',
                    tokenizer=tokenizer,
                    event_types=seen_event_types
                )
                print(f'==== {len(sub_task_dataset)} {split} samples in Task {task_id} ====')

        dataloader = DataLoader(
            sub_task_dataset,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            num_workers=4
        )
        task_dataloaders.append(dataloader)
    
    return task_dataloaders


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = MavenDataset(tokenizer=tokenizer)
    print(len(dataset))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 若模型无 pad_token，用 eos_token 代替

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    for batch in dataloader:
        print(batch)
        break
