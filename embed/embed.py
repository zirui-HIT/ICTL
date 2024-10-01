import os
import sys
import json
import pickle
import random
import argparse
import numpy as np

from typing import List, Dict, Any


sys.path.append('.')
random.seed(16)


def batch_data(data_list: List[str], batch_size: int) -> List[List[str]]:
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:])
    return batch_data


if __name__ == '__main__':
    from utils.embedder import FlagICLEmbedder

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--task_dir', type=str)
    # parser.add_argument('--task_file', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument(
        '--order', type=str, choices=['random', 'reverse', 'normal'], default='normal')
    args = parser.parse_args()

    embedder = FlagICLEmbedder(args.model_name_or_path)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    task_names_used = [f.split('.')[0] for f in os.listdir(
        args.task_dir) if f.endswith('.json')]
    task_names_used = sorted(
        task_names_used, reverse=(args.order == 'reverse'))
    if args.order == 'random':
        random.shuffle(task_names_used)

    for file_name in task_names_used:
        load_file = os.path.join(args.task_dir, file_name + '.json')
        dump_file = os.path.join(args.save_dir, file_name + '.pkl')
        if not os.path.exists(load_file):
            print(f'{file_name} not found.')
            continue
        if os.path.exists(dump_file):
            print(f'{file_name} already embedded.')
            continue

        print(f'Embedding {file_name}...')
        with open(load_file, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)

        if len(data['Definition']) > 1:
            print('Definition:', data['Definition'])
        prompts: List[str] = ['\n'.join(data['Definition'])]
        idxs: List[str] = [None]
        for instance in data['Instances']:
            if not isinstance(instance['output'], list):
                instance['output'] = [instance['output']]
            if not isinstance(data['Definition'], list):
                data['Definition'] = [data['Definition']]
            prompts.extend(['\n'.join([
                str(data['Definition'][0]),
                str(instance['input']),
                f"{instance['explanation']}\nSo the answer is: {o}" if 'explanation' in instance else str(o),
            ]) for o in instance['output']])
            idxs.extend([instance['id']] * len(instance['output']))
        print("---")
        print(prompts[0])
        print("---")
        print(prompts[-1])
        print("---")

        examples = ['\n'.join([
            str(data['Definition'][0]),
            str(e['input']),
            f"{e['explanation']}\nSo the answer is: {e['output']}"
        ]) for e in data['Positive Examples']]
        information = {
            "instruction": "Given a demonstration, retrieve the similar demonstration of the given demonstration.",
            "examples": [{
                "instruct": "Given a demonstration, retrieve the similar demonstration of the given demonstration.",
                "query": examples[2*i],
                "response": examples[2*i+1]
            } for i in range(len(examples) // 2)]
        }
        embeddings = embedder.embed(prompts, config, information)
        results: Dict[str, Any] = {
            "Definition": np.array(embeddings[0]),
            "Instances": [{
                "id": idx,
                "prompt": np.array(embedding)
            } for idx, embedding in zip(idxs[1:], embeddings[1:])],
        }

        with open(dump_file, 'wb') as f:
            pickle.dump(results, f)
