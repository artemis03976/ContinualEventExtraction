import json
from utils import compute_acc, compute_event_f1


def evaluate(response_file_path, label_file_path):
    # Step 1: 加载 label 文件，建立 custom_id 到 label 的映射
    label_map = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label_entry = json.loads(line)
            cid = label_entry.get('custom_id')
            label = json.loads(label_entry.get('label'))
            task_id = label_entry.get('task_id')
            if cid is not None:
                label_map[cid] = {'label': label, 'task_id': task_id}

    total_results = {task_id: [] for task_id in range(1, 6)}
    # Step 2: 读取模型响应文件并处理
    with open(response_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry.get('custom_id')
            message_content = (
                entry.get('response', {})
                     .get('body', {})
                     .get('choices', [{}])[0]
                     .get('message', {})
                     .get('content', '')
            )
            try:
                prediction = json.loads(message_content)
            except json.JSONDecodeError:
                print
                prediction = []

            label = label_map.get(custom_id)
            total_results[label['task_id']].append((prediction, label['label']))

    task_acc = 0.0
    for task_id, results in total_results.items():
        task_f1_score = compute_event_f1(results)
        print(f"Task {task_id} F1: {task_f1_score['f1']:.4f}")

        task_acc += compute_acc(results)
    
    print(f"Acc: {task_acc / len(total_results):.4f}")


if __name__ == '__main__':
    response_file_path = './checkpoints/zero-shot/results_ere.jsonl'
    label_file_path = './data/ERE/labels.jsonl'
    evaluate(response_file_path, label_file_path)
