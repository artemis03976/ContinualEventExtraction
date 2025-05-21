from data import MavenDataset, ACEDataset, EREDataset, split_tasks_by_event_types, get_event_types
import argparse
from prompt_template import zero_shot_prompt
import json
import os


def get_args():
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    return args


def load_data(args, seen_event_types):
    if args.dataset == 'maven':
        sub_task_dataset = MavenDataset(
            root=args.data_root,
            phase='trigger',
            split='valid',
            event_types=seen_event_types
        )
    elif args.dataset == 'ace2005':
        sub_task_dataset = ACEDataset(
            root=args.data_root,
            phase='trigger',
            split='dev',
            event_types=seen_event_types
        )
    elif args.dataset == 'ere':
        sub_task_dataset = EREDataset(
            root=args.data_root,
            phase='trigger',
            split='valid',
            event_types=seen_event_types
        )

    return sub_task_dataset.trigger_data


def build_prompt(data, seen_event_types):
    messages = [
        {"role": "system", "content": zero_shot_prompt["system"]},
        {"role": "user", "content": zero_shot_prompt["user"].format(
            input_text=data['input_text'],
            event_types=seen_event_types,
        )},
    ]

    return messages


def build_request(data_id, messages):
    return {
        "custom_id": data_id, 
        "body": {
            "messages": messages,
            "max_tokens": 512
        }
    }


def main():
    args = get_args()
    event_types = get_event_types(args)
    event_type_groups = split_tasks_by_event_types(event_types=event_types, n_tasks=args.n_tasks, strategy=args.strategy)
    
    prompt_label_pairs = []

    for task_id, event_type_per_task in enumerate(event_type_groups):
        datas = load_data(args, event_type_per_task)
        for data in datas:
            prompt = build_prompt(data, event_type_per_task)
            label = data['target_output']
            prompt_label_pairs.append((prompt, label, task_id))
    
    with open(os.path.join(args.data_root, 'requests.jsonl'), 'w', encoding='utf-8') as rf, \
        open(os.path.join(args.data_root, 'labels.jsonl'), 'w', encoding='utf-8') as lf:

        for index, (prompt, label, task_id) in enumerate(prompt_label_pairs):
            # 写入 prompt 数据
            rf.write(json.dumps(build_request(f"request_{index}", prompt), ensure_ascii=False) + '\n')
            # 写入 label 数据
            lf.write(json.dumps({"custom_id": f"request_{index}", "task_id": task_id + 1, "label": label}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
