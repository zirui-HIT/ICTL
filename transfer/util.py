import math
import random
import numpy as np

from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from sample.util import generateTheta, sWasserstein


def optimize_transfer_random(
    instruction: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    number: int,
    dim: int,
    epsilon: float = 1e-4,
    theta: np.ndarray = None,
    max_retry_times: int = 1024
) -> Tuple[List[str], float]:
    if not (theta is not None):
        theta = generateTheta(dim, instruction.shape[1])

    step = 0
    best_score = float('inf')
    best_idxs: List[List[int]] = []
    while step < max_retry_times:
        idxs_selected = random.sample(range(len(targets)), number)
        targets_selected = targets[idxs_selected]
        dataset_score = sWasserstein(targets_selected, sources, theta)
        instruction_score = sWasserstein(targets_selected, instruction, theta)
        current_score = dataset_score + instruction_score

        if current_score < best_score:
            best_score = current_score
            best_idxs = idxs_selected
            step = 0
        elif abs(current_score-best_score) < epsilon:
            break
        step += 1

    return best_idxs, best_score


def optimize_transfer_random_multiple(

    instruction: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    number: int,
    dim: int,
    epsilon: float = 1e-4,
    theta: np.ndarray = None,
    max_retry_times: int = 512,
    repeat: int = 10
):
    if theta is None:
        theta = generateTheta(dim, instruction.shape[1])

    best_score = float('inf')
    best_idxs = []
    for _ in range(repeat):
        idxs, score = optimize_transfer_random(
            instruction, sources, targets, number, dim, epsilon, theta, max_retry_times)
        if score < best_score:
            best_score = score
            best_idxs = idxs

    return best_idxs, best_score


def optimize_transfer_anneal(
    instruction: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    number: int,
    dim: int,
    epsilon: float = 1e-4,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.99,
    perturbation_size: int = 1,
    min_temp: float = 1e-7,
    reheat_factor: float = 1.05,
    tabu_size: int = 10
) -> Tuple[List[int], float]:

    def acceptance_probability(old_score, new_score, temperature):
        return 1.0 if new_score < old_score else math.exp((old_score - new_score) / temperature)

    def perturb_solution(idxs: List[int], tabu_list: set) -> List[int]:
        new_idxs = idxs[:]
        for _ in range(perturbation_size):
            replace_idx = random.randint(0, len(new_idxs) - 1)
            candidates = [i for i in range(
                len(targets)) if i not in new_idxs and i not in tabu_list]
            if candidates:
                new_candidate = random.choice(candidates)
                new_idxs[replace_idx] = new_candidate
        return new_idxs

    def dynamic_cooling_rate(temperature, score_diff):
        return cooling_rate * reheat_factor if score_diff < epsilon else cooling_rate

    def compute_score(new_idxs):
        return (sWasserstein(targets[new_idxs], sources, theta) +
                sWasserstein(targets[new_idxs], instruction, theta))

    theta = generateTheta(dim, instruction.shape[1])
    current_idxs, current_score = optimize_transfer_random(
        instruction, sources, targets, number, dim, epsilon, theta=theta)
    best_score = current_score
    best_idxs = current_idxs
    temperature = initial_temperature
    tabu_list = set()

    while temperature > min_temp:
        new_idxs_list = [perturb_solution(current_idxs, tabu_list)
                         for _ in range(cpu_count() // 2)]

        scores = [compute_score(new_idxs) for new_idxs in new_idxs_list]
        min_score_idx = np.argmin(scores)
        new_score = scores[min_score_idx]
        new_idxs = new_idxs_list[min_score_idx]

        score_diff = abs(current_score - new_score)

        if acceptance_probability(current_score, new_score, temperature) > random.random():
            current_idxs = new_idxs
            current_score = new_score

        if current_score < best_score:
            best_score = current_score
            best_idxs = current_idxs

        temperature *= dynamic_cooling_rate(temperature, score_diff)

        if len(tabu_list) > tabu_size:
            tabu_list.pop()

        tabu_list.add(tuple(best_idxs))

        if score_diff < epsilon:
            break

    return best_idxs, best_score
