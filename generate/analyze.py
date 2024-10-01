import json
import argparse

from typing import List, Dict


def extract_idx(name: str) -> int:
    return int(name.split('_')[0].split('task')[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--category_file', type=str)
    parser.add_argument('--task_file', type=str)
    args = parser.parse_args()

    with open(args.pred_file, 'r', encoding='utf-8') as f:
        data: Dict[str, float] = json.load(f)
    with open(args.category_file, 'r', encoding='utf-8') as f:
        category: Dict[str, List[int]] = json.load(f)
    with open(args.task_file, 'r', encoding='utf-8') as f:
        tasks: List[str] = f.read().splitlines()

    score_map: Dict[int, float] = {}
    for name, score in data.items():
        if 'task' not in name or 'rougeL' not in name:
            continue
        task_name = name.split('_default_track')[0].split('for_task')[-1]
        task_idx = extract_idx(task_name)
        score_map[task_idx] = score

    task_map: Dict[str, str] = {}
    total_map: Dict[int, float] = {}
    count_map: Dict[int, int] = {}
    for name in tasks:
        task_idx = extract_idx(name)
        flag = False
        for c, x in category.items():
            if task_idx in x:
                flag = True
                count_map[c] = count_map.get(c, 0) + 1
                total_map[c] = total_map.get(c, 0) + score_map[task_idx]
                break
        if not flag:
            task_map[name] = None
        else:
            task_map[name] = c
    # print(len(task_map))
    # print(json.dumps(task_map, indent=4))
    count_map = {k: v for k, v in sorted(
        count_map.items(), key=lambda x: x[0])}
    print(json.dumps({k: total_map[k] / count_map[k]
          for k, v in count_map.items()}, indent=4))
