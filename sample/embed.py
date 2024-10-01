import os
import json
import pickle
import argparse
import numpy as np

from typing import List, Dict
from tqdm.contrib.concurrent import process_map


def process_test_task(test_name: str) -> None:
    load_file = os.path.join(args.sample_dir, test_name + '.json')
    dump_file = os.path.join(args.sample_dir, test_name + '.pkl')
    if not os.path.exists(load_file):
        print(f'{test_name} does not exist')
        return
    if os.path.exists(dump_file):
        print(f'{test_name} already exists')
        return

    with open(os.path.join(args.sample_dir, test_name + '.json'), 'r') as f:
        task: Dict[str, List[str]] = json.load(f)

    embeddings = []
    for task_name, example_ids in task.items():
        with open(os.path.join(args.embed_dir, task_name + '.pkl'), 'rb') as f:
            task_embed = pickle.load(f)
        embeddings.extend(
            [x['prompt']
                for x in task_embed['Instances'] if x['id'] in example_ids]
        )

    with open(os.path.join(args.sample_dir, test_name + '.pkl'), 'wb') as f:
        pickle.dump(np.array(embeddings), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_tasks', type=str)
    parser.add_argument('--sample_dir', type=str)
    parser.add_argument('--embed_dir', type=str)
    args = parser.parse_args()

    # Read the list of test tasks
    with open(args.test_tasks, 'r') as f:
        test_tasks = f.read().splitlines()

    # Use process_map to parallelize and show a progress bar
    process_map(process_test_task, test_tasks,
                max_workers=os.cpu_count() // 2, chunksize=1)
