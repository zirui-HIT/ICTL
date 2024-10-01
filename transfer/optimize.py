import os
import sys
import json
import pickle
import random
import argparse
import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


random.seed(42)
np.random.seed(42)
sys.path.append('.')


def process_task(name):
    load_file = os.path.join(args.transfer_dir, name + '.pkl')
    dump_file = os.path.join(args.dump_dir, name + '.json')
    if os.path.exists(dump_file):
        print(f'{dump_file} already exists. Skip.')
        return
    if not os.path.exists(load_file):
        print(f'{load_file} not found. Skip.')
        return

    with open(os.path.join(args.source_embed_dir, name + '.pkl'), 'rb') as f:
        source_dataset = pickle.load(f)
    with open(load_file, 'rb') as f:
        target_dataset = pickle.load(f)

    if len(target_dataset['Instances']) <= args.sample_scale:
        idx_selected = list(range(len(target_dataset['Instances'])))
    else:
        idx_selected, _ = optimize_transfer_anneal(
            target_dataset['Definition'].reshape(1, -1),
            source_dataset,
            np.array([x['prompt'] for x in target_dataset['Instances']]),
            args.sample_scale,
            2, epsilon=1e-4, cooling_rate=0.99
        )

    with open(os.path.join(args.transfer_dir, name + '.json'), 'r', encoding='utf-8') as f:
        target_dataset = json.load(f)

    target_dataset = [target_dataset[idx] for idx in idx_selected]

    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(target_dataset, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from util import optimize_transfer_anneal

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--transfer_dir', type=str)
    parser.add_argument('--dump_dir', type=str)
    parser.add_argument('--source_embed_dir', type=str)
    parser.add_argument(
        '--order', type=str, choices=['random', 'reverse', 'normal'], default='normal')
    parser.add_argument('--sample_scale', type=int, default=2048)
    args = parser.parse_args()

    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_task_names = f.read().splitlines()
    test_task_names = sorted(
        test_task_names, reverse=(args.order == 'reverse'))
    if args.order == 'random':
        random.shuffle(test_task_names)

    # Use process_map to parallelize the task processing
    process_map(process_task, test_task_names,
                max_workers=os.cpu_count() // 2, chunksize=1)
    # for name in tqdm(test_task_names):
    #     process_task(name)
