import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from typing import List, Dict, Any
from tqdm.contrib.concurrent import process_map


random.seed(42)
np.random.seed(42)
sys.path.append('.')


def process_test_name(test_name: str) -> None:
    dump_file = os.path.join(args.dump_dir, f"{test_name}.json")
    if os.path.exists(dump_file):
        print(f"Skip {test_name}")
        return

    try:
        with open(os.path.join(args.embed_dir, test_name + '.pkl'), 'rb') as f:
            test_dataset = pickle.load(f)
    except Exception as e:
        print(f"Error in {test_name}: {e}")
        return
    train_datasets = []
    for train_name in task_map[test_name]:
        try:
            with open(os.path.join(args.embed_dir, train_name + '.pkl'), 'rb') as f:
                train_datasets.append(pickle.load(f))
        except Exception as e:
            print(f"Error in {train_name}: {e}")
            return
    print(f"Data Loaded for {test_name}")

    used_idx: Dict[str, List[str]] = {}
    if args.sample_scale >= sum([len(dataset['Instances']) for dataset in train_datasets]):
        idxs_selected = [f"{i}.{j}" for i,
                         dataset in enumerate(train_datasets) for j in range(len(dataset['Instances']))]
        score = 0
    else:
        idxs_selected, score = optimize_sample_random(
            test_dataset['Definition'].reshape(1, -1), [{
                'dataset': np.stack([x['prompt'] for x in train_dataset['Instances']]),
                'instruction': train_dataset['Definition'].reshape(1, -1)
            } for train_dataset in train_datasets], args.sample_scale, 2)

    for i, dataset in enumerate(train_datasets):
        for j, instance in enumerate(dataset['Instances']):
            if f"{i}.{j}" in idxs_selected:
                train_name = task_map[test_name][i]
                if train_name not in used_idx:
                    used_idx[train_name] = []
                used_idx[train_name].append(instance['id'])
    # print(f"Optimized for {test_name} with score {score}")

    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(used_idx, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from sample.util import optimize_sample_random

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--task_map', type=str)
    parser.add_argument('--dump_dir', type=str)
    parser.add_argument('--embed_dir', type=str)
    parser.add_argument('--sample_scale', type=int)
    parser.add_argument(
        '--order', type=str, choices=['random', 'reverse', 'normal'], default='normal')
    args = parser.parse_args()

    test_task_names = sorted([line.strip() for line in open(
        args.test_file, 'r', encoding='utf-8')], reverse=(args.order == 'reverse'))
    if args.order == 'random':
        random.shuffle(test_task_names)
    with open(args.task_map, 'r', encoding='utf-8') as f:
        task_map: Dict[str, List[str]] = json.load(f)

    # for task in tqdm(test_task_names):
    #     process_test_name(task)
    process_map(process_test_name, test_task_names,
                max_workers=mp.cpu_count()//2, chunksize=1)
