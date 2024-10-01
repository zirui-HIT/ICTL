import os
import json
import argparse

from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', type=str,
                    default='./generate/result/sampling_strategy/ours')
parser.add_argument('--data_file', type=str)
args = parser.parse_args()


def merge(data_file_names: List[str], load_dir: str) -> None:
    all_data = []
    for file in data_file_names:
        load_file = os.path.join(load_dir, file + '.json')
        if not os.path.exists(load_file):
            continue
        with open(load_file, 'r', encoding='utf-8') as f:
            all_data.extend(json.load(f))
    all_data = [{
        'id': d['id'],
        'prediction': d['prediction']
    } for d in all_data]
    with open(os.path.join(load_dir, 'prediction.jsonl'), 'w', encoding='utf-8') as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data_file_names = [x.strip() for x in f if x.strip()]
    merge(data_file_names, args.load_dir)
    print(f"Merge {args.load_dir} done.")
