import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from prompt_template import base_model_prompt
from functools import partial
import random
from utils import get_mask_len


def split_tasks_by_event_types(args, event_types):
    """
    Split event types for task sequences
    Args:
        event_types: list of all event types
        n_tasks: number of tasks
        strategy: split strategy ('sequential'|'random'|'balanced'|'clustered')
    """
    if args.strategy == 'sequential':
        types_per_task = len(event_types) // args.n_tasks
        return [
            event_types[i * types_per_task : (i + 1) * types_per_task] 
            for i in range(args.n_tasks)
        ]

    elif args.strategy == 'random':
        shuffled = random.sample(event_types, len(event_types))
        return [shuffled[i::args.n_tasks] for i in range(args.n_tasks)]
    
    elif args.strategy == 'balanced':
        # 按数据量平衡划分（需预统计类型分布）
        # 此处需实现数据量统计逻辑
        raise NotImplementedError
    
    elif args.strategy == 'clustered':
        # 基于语义聚类划分（需类型嵌入表示）
        # 此处需实现聚类算法
        raise NotImplementedError

    elif args.strategy == 'local':
        with  open(os.path.join(args.data_root, 'event_type_groups.json'), 'r', encoding='utf-8') as f:
            event_types = json.load(f)
        return event_types
    
    else:
        raise ValueError(f"未知划分策略: {args.strategy}")


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
        
        elif args.dataset == 'ace2005':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    for event in doc.get('events', []):
                        event_types.add(event['event_type'])
        
        elif args.dataset == 'ere':
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
    def __init__(
            self, 
            root='./data/', 
            phase='trigger', 
            split='train', 
            tokenizer=None,
            task_id=1, 
            n_tasks=5,
            event_types=None, 
            load_generation=False
        ):

        self.root = root
        self.phase = phase
        self.split = split
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.event_types = event_types
        self.load_generation = load_generation

        # data for trigger classification
        self.trigger_data = None
        # data for argument classification
        self.argument_data = None
        
        # 生成唯一缓存标识（基于任务划分参数）
        self.cache_key = f"{self.phase}_{self.split}_task_{task_id}_of_{n_tasks}"
        self.cache_path = os.path.join(self.root, f"cache/{self.cache_key}.json")

        # 直接加载预处理后的任务数据
        if not os.path.exists(self.cache_path):
            self._preprocess_and_cache()
        self._load_cached_data()
            
    def _load_raw_data(self):  
        # load raw data
        self.raw_data = []
        file_path = os.path.join(self.root, "{}.jsonl".format(self.split))
        with open(file_path, "r", encoding="utf-8") as f:
            for data in f:
                docs = json.loads(data)
                self.raw_data.append(docs)
        
        # construct corresponding dataset
        if self.phase == 'trigger':
            self.raw_trigger_data = self.construct_trigger_dataset()
        elif self.phase == 'argument':
            self.raw_argument_data = self.construct_argument_dataset()
    
    def _load_cached_data(self):
        """从缓存加载预处理后的任务数据"""
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.phase == 'trigger':
            self.trigger_data = data
        elif self.phase == 'argument':
            self.argument_data = data

    def _preprocess_and_cache(self):
        """预处理并缓存任务数据"""
        # 读取原始数据（需实现具体逻辑）
        self._load_raw_data()  
        
        # 划分数据（含无事件样本分配）
        data = self._split_data_by_task()
          
        # 保存缓存
        os.makedirs(os.path.join(self.root, "cache"), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _split_data_by_task(self):
        """核心划分逻辑"""
        task_data = []
        none_samples = []
        
        # 收集有效事件样本
        for item in self.raw_trigger_data:
            events = json.loads(item['target_output'])
            valid_events = []
            for event in events:
                if event['event_type'] in self.event_types:
                    valid_events.append(event)
            
            if len(valid_events) != 0:
                new_item = item.copy()
                new_item['target_output'] = json.dumps(valid_events)
                task_data.append(new_item)
        
        # 收集无事件样本
        for item in self.raw_trigger_data:
            events = json.loads(item['target_output'])
            if all(event['event_type'] == 'None' for event in events):
                none_samples.append(item)
        
        # 均匀分配无事件样本
        per_task = len(none_samples) // self.n_tasks
        start = (self.task_id - 1) * per_task
        end = start + per_task if self.task_id != self.n_tasks else None
        assigned_none = none_samples[start:end]
        
        return task_data + assigned_none

    def construct_trigger_dataset(self, force_rebuild=False):
        raise NotImplementedError

    def construct_argument_dataset(self, force_rebuild=False):
        raise NotImplementedError

    def __len__(self):
        if self.phase == 'trigger':
            return len(self.trigger_data)
        elif self.phase == 'argument':
            return len(self.argument_data)

    def build_training_prompt(self, input_text, target_output):
        messages = [
            {"role": "system", "content": base_model_prompt["system"]},
            {"role": "user", "content": base_model_prompt["user"].format(input_text=input_text)},
            {"role": "assistant", "content": base_model_prompt["assistant_prefix"]}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        mask_len = get_mask_len(prompt_ids, self.tokenizer)

        # 1. Construct input ids
        messages[-1]["content"] += target_output
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = self.tokenizer(
            full_prompt,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )["input_ids"][0]

        # 2. Construct labels
        labels = input_ids.clone()
        labels[:mask_len] = -100 

        # 3. Construct text ids
        text_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        attention_mask = torch.ones_like(input_ids)

        return text_ids, input_ids, attention_mask, labels

    def build_generation_prompt(self, input_text):
        # 构建不闭合的 assistant 消息，避免提前终止
        messages = [
            {"role": "system", "content": base_model_prompt["system"]},
            {"role": "user", "content": base_model_prompt["user"].format(input_text=input_text)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt += "<|im_start|>assistant\n" + base_model_prompt["assistant_prefix"]

        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        attention_mask = torch.ones_like(input_ids)
        text_ids = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        return text_ids, input_ids, attention_mask

    def __getitem__(self, idx):
        if self.phase == 'trigger':
            input_text, target_output = self.trigger_data[idx]['input_text'], self.trigger_data[idx]['target_output']
        elif self.phase == 'argument':
            input_text, target_output = self.argument_data[idx]['input_text'], self.argument_data[idx]['target_output']

        if not self.load_generation:
            text_ids, input_ids, attention_mask, labels = self.build_training_prompt(input_text, target_output)
        else:
            text_ids, input_ids, attention_mask = self.build_generation_prompt(input_text)
            labels = target_output
            
        return {
            'text_ids': text_ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels,
        }


class MavenDataset(ContinualEventExtractionDataset):
    def __init__(
            self, 
            root='./data/MAVEN', 
            phase='trigger', 
            split='train', 
            tokenizer=None, 
            task_id=1, 
            n_tasks=5,
            event_types=None, 
            load_generation=False
        ):
        super().__init__(root, phase, split, tokenizer, task_id, n_tasks, event_types, load_generation)

    def construct_trigger_dataset(self, force_rebuild=False):
        cache_path = os.path.join(self.root, f'{self.phase}_{self.split}.json')
        if not force_rebuild and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_trigger_data = json.load(f)
            return raw_trigger_data

        raw_trigger_data = []
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
                if len(all_events[i]) == 0:
                    all_events[i].append({
                        'trigger_word': 'None',
                        'event_type': 'None',
                        'span': 'None'
                    })
                raw_trigger_data.append({
                    'input_text': text,
                    'target_output': json.dumps(all_events[i])
                })
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(raw_trigger_data, f, ensure_ascii=False, indent=4)

        return raw_trigger_data


class EREDataset(ContinualEventExtractionDataset):
    def __init__(
            self, 
            root='./data/ERE', 
            phase='trigger', 
            split='train', 
            tokenizer=None, 
            task_id=1, 
            n_tasks=5,
            event_types=None, 
            load_generation=False
        ):
        super().__init__(root, phase, split, tokenizer, task_id, n_tasks, event_types, load_generation)

    def construct_trigger_dataset(self, force_rebuild=False):
        cache_path = os.path.join(self.root, f'{self.phase}_{self.split}.json')
        if not force_rebuild and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_trigger_data = json.load(f)
            return raw_trigger_data

        raw_trigger_data = []
        for docs in self.raw_data:
            # get event text
            text = docs['text']

            # get events and classify by sentence id
            all_events = [] 
            for event in docs['events']:
                for mention in event['triggers']:
                    # discard useless info
                    processed_event = {
                        'trigger_word': mention['trigger_word'],
                        'event_type': event['type'],
                        'span': mention['position']
                    }

                    all_events.append(processed_event)

            if len(all_events) == 0:
                all_events.append({
                    'trigger_word': 'None',
                    'event_type': 'None',
                    'span': 'None'
                })

            raw_trigger_data.append({
                'input_text': text,
                'target_output': json.dumps(all_events)
            })
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(raw_trigger_data, f, ensure_ascii=False, indent=4)

        return raw_trigger_data


class ACEDataset(ContinualEventExtractionDataset):
    def __init__(
            self, 
            root='./data/ACE2005-en', 
            phase='trigger', 
            split='train', 
            tokenizer=None, 
            task_id=1, 
            n_tasks=5,
            event_types=None, 
            load_generation=False
        ):
        super().__init__(root, phase, split, tokenizer, task_id, n_tasks, event_types, load_generation)

    def construct_trigger_dataset(self, force_rebuild=False):
        cache_path = os.path.join(self.root, f'{self.phase}_{self.split}.json')
        if not force_rebuild and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw_trigger_data = json.load(f)
            return raw_trigger_data

        raw_trigger_data = []
        for docs in self.raw_data:
            # get event text
            text = docs['sentence']

            # get events and classify by sentence id
            all_events = [] 
            for event in docs['events']:
                # discard useless info
                processed_event = {
                    'trigger_word': event['trigger']['text'],
                    'event_type': event['event_type'],
                    'span': [event['trigger']['start'], event['trigger']['end']]
                }

                all_events.append(processed_event)

            if len(all_events) == 0:
                all_events.append({
                    'trigger_word': 'None',
                    'event_type': 'None',
                    'span': 'None'
                })

            raw_trigger_data.append({
                'input_text': text,
                'target_output': json.dumps(all_events)
            })
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(raw_trigger_data, f, ensure_ascii=False, indent=4)
        
        return raw_trigger_data


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
        padded_seq = torch.cat([item["text_ids"], torch.tensor(padding, dtype=torch.long, device=item["text_ids"].device)])
        padded_batch['text_ids'].append(padded_seq)
    
        # 填充 input_ids
        padding = [tokenizer.pad_token_id] * (max_input_len - len(item["input_ids"]))
        padded_seq = torch.cat([item["input_ids"], torch.tensor(padding, dtype=torch.long, device=item["input_ids"].device)])
        padded_batch['input_ids'].append(padded_seq)
    
        # 填充 attention_mask
        padding = [0] * (max_input_len - len(item["input_ids"]))
        padded_seq = torch.cat([item["attention_mask"], torch.tensor(padding, dtype=torch.long, device=item["attention_mask"].device)])
        padded_batch['attention_mask'].append(padded_seq)

        if isinstance(item["labels"], torch.Tensor):
            # 填充 labels
            padding = [-100] * (max_input_len - len(item["labels"]))
            padded_seq = torch.cat([item["labels"], torch.tensor(padding, dtype=torch.long, device=item["labels"].device)])
            padded_batch['labels'].append(padded_seq)
        else:
            padded_batch['labels'].append(item["labels"])

    if isinstance(padded_batch['labels'][0], torch.Tensor):
        return {
            "text_ids": torch.stack(padded_batch['text_ids']),
            "input_ids": torch.stack(padded_batch['input_ids']),
            "attention_mask": torch.stack(padded_batch['attention_mask']),
            "labels": torch.stack(padded_batch['labels']) ,
        }
    else:
        return {
            "text_ids": torch.stack(padded_batch['text_ids']),
            "input_ids": torch.stack(padded_batch['input_ids']),
            "attention_mask": torch.stack(padded_batch['attention_mask']),
            "labels": padded_batch['labels'],
        }


def get_dataloader(args, event_type_groups, phase='trigger', split='train', tokenizer=None):
    task_dataloaders = []

    for task_id, event_type_per_task in enumerate(event_type_groups):
        if split == 'train':
            if args.dataset == 'maven':
                sub_task_dataset = MavenDataset(
                    root=args.data_root,
                    phase=phase,
                    split=split,
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ace2005':
                sub_task_dataset = ACEDataset(
                    root=args.data_root,
                    phase=phase,
                    split=split,
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ere':
                sub_task_dataset = EREDataset(
                    root=args.data_root,
                    phase=phase,
                    split=split,
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            
            print(f'==== {len(sub_task_dataset)} {split} samples in Task {task_id + 1} ====')

        elif split == 'dev':
            if args.dataset == 'maven':
                sub_task_dataset = MavenDataset(
                    root=args.data_root,
                    phase=phase,
                    split='valid',
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ace2005':
                sub_task_dataset = ACEDataset(
                    root=args.data_root,
                    phase=phase,
                    split=split,
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ere':
                sub_task_dataset = EREDataset(
                    root=args.data_root,
                    phase=phase,
                    split='valid',
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            print(f'==== {len(sub_task_dataset)} {split} samples in Task {task_id + 1} ====')
    
        elif split == 'test':
            if args.dataset == 'maven':
                sub_task_dataset = MavenDataset(
                    root=args.data_root,
                    phase=phase,
                    split='valid',
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    load_generation=True,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ace2005':
                sub_task_dataset = ACEDataset(
                    root=args.data_root,
                    phase=phase,
                    split='dev',
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    load_generation=True,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            elif args.dataset == 'ere':
                sub_task_dataset = EREDataset(
                    root=args.data_root,
                    phase=phase,
                    split='valid',
                    tokenizer=tokenizer,
                    event_types=event_type_per_task,
                    load_generation=True,
                    task_id=task_id + 1,
                    n_tasks=args.n_tasks
                )
            print(f'==== {len(sub_task_dataset)} {split} samples in Task {task_id + 1} ====')

        dataloader = DataLoader(
            sub_task_dataset,
            batch_size=(args.batch_size if split != 'test' else 1),
            shuffle=(split == 'train'),
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            num_workers=4
        )
        task_dataloaders.append(dataloader)
    
    return task_dataloaders


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("../hf_cache/hub/Llama-3.1-8B-Instruct")
    dataset = MavenDataset(tokenizer=tokenizer)
    print(len(dataset))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    for batch in dataloader:
        print(batch)
        break
