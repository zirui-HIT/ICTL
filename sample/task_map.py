import os
import json
import pickle
import numpy as np
import multiprocessing as mp

from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map

SCALE_LIMIT = 32
EMBEDDING_DIR = "./embed/embedding/Super-NI"


def rank_by_cosine_similarity(source: List[np.ndarray], target: List[np.ndarray]) -> List[List[int]]:
    source_matrix = np.stack(source)
    target_matrix = np.stack(target)
    source_norms = np.linalg.norm(source_matrix, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)
    cosine_sim_matrix = (target_matrix @ source_matrix.T) / \
        (target_norms @ source_norms.T)
    ranked_indices = np.argsort(-cosine_sim_matrix, axis=1)
    return ranked_indices.tolist()


def load_task(args: Tuple[str, str]) -> Dict[str, Any]:
    task, embedding_dir = args
    with open(os.path.join(embedding_dir, f"{task}.pkl"), "rb") as f:
        dataset = pickle.load(f)
        return {
            "task": task,
            "introduction": dataset['Definition'],
            "scale": len(dataset['Instances'])
        }


def load_data(task_file: str, embedding_dir: str) -> List[Dict[str, Any]]:
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = [task.strip() for task in f]
    args = [(task, embedding_dir) for task in tasks]

    datasets = process_map(
        load_task,
        args,
        max_workers=mp.cpu_count(),
        chunksize=1,
        desc="Loading tasks",
        unit="task"
    )

    return datasets


if __name__ == '__main__':
    test_datasets = load_data(
        "./dataset/Super-NI/splits/default/test_tasks.txt", EMBEDDING_DIR)
    train_datasets = load_data(
        "./dataset/Super-NI/splits/default/train_tasks.txt", EMBEDDING_DIR)

    train_idx_ranked = rank_by_cosine_similarity([d['introduction'] for d in train_datasets], [
                                                 d['introduction'] for d in test_datasets])
    results: Dict[str, List[str]] = {}
    for d, idxs in zip(test_datasets, train_idx_ranked):
        results[d['task']] = []
        count = 0
        for idx in idxs:
            if train_datasets[idx]['task'] not in results[d['task']]:
                results[d['task']].append(train_datasets[idx]['task'])
                count += 1
            if count > SCALE_LIMIT:
                break
    with open('./sample/result/Super-NI/sampling_strategy/large/task_map.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
