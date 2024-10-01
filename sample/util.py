import random
import numpy as np

from typing import List, Dict, Tuple


random.seed(42)
np.random.seed(42)


def generateTheta(L: int, endim: int) -> np.ndarray:
    """
    Generates a matrix of random vectors, each normalized to have a unit length.

    Parameters:
    L (int): The number of random vectors to generate.
    endim (int): The dimensionality of each random vector.

    Returns:
    np.ndarray: A matrix of shape (L, endim) where each row is a normalized vector.
    """
    theta_ = np.random.normal(size=(L, endim))
    for l in range(L):
        theta_[l, :] = theta_[l, :]/np.sqrt(np.sum(theta_[l, :]**2))
    return theta_


def sWasserstein(source: np.ndarray, target: np.ndarray, theta: np.ndarray) -> float:
    def oneDWasserstein(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Sort the input arrays
        psort = np.sort(p, axis=0)
        qsort = np.sort(q, axis=0)

        # Compute the cumulative sum scaled by the maximum length
        n_p, n_q = p.shape[0], q.shape[0]
        max_n = max(n_p, n_q)

        # Calculate cumulative sums directly
        pcum = max_n * np.cumsum(psort) / n_p
        qcum = max_n * np.cumsum(qsort) / n_q

        # Use linspace to get the minimum number of indices
        indices = np.arange(min(n_p, n_q))

        # Calculate the differences using vectorized operations
        phat_diff = np.diff(np.concatenate(([0], pcum[indices])))
        qhat_diff = np.diff(np.concatenate(([0], qcum[indices])))

        # Compute the Wasserstein distance
        w2 = np.mean((phat_diff - qhat_diff) ** 2)
        return w2

    source = np.dot(source, theta.T)
    target = np.dot(target, theta.T)
    return np.mean(oneDWasserstein(source, target))


def optimize_sample_random(
    instruction: np.ndarray,
    targets: List[Dict[str, np.ndarray]],
    number: int,
    dim: int,
    epsilon: float = 1e-4,
    max_steps: int = 512,
    max_steps_total: int = 16384,
    theta: np.ndarray = None
) -> Tuple[List[str], float]:
    if theta is None:
        theta = generateTheta(dim, targets[0]['dataset'].shape[1])

    idxs: List[str] = [f"{i}.{j}" for i in range(
        len(targets)) for j in range(len(targets[i]['dataset']))]

    best_score = float('inf')
    best_idxs: List[str] = []

    step = 0
    step_total = 0
    while step < max_steps:
        idxs_selected = random.sample(idxs, number)
        current_score = 0
        for i in range(len(targets)):
            idxs_selected_current: List[int] = [
                int(idx.split('.')[-1]) for idx in idxs_selected if idx.startswith(f"{i}.")]
            if not idxs_selected_current:
                continue

            dataset_score = sWasserstein(
                targets[i]['dataset'][idxs_selected_current], instruction, theta)
            instruction_score = sWasserstein(
                targets[i]['instruction'], instruction, theta)
            current_score += len(idxs_selected_current) / \
                number * (6 * dataset_score + instruction_score)

        if current_score < best_score:
            # print(
            #     f"Update best score in step {step_total}: {best_score} -> {current_score}")
            best_score = current_score
            best_idxs = idxs_selected
            step = 0
        elif abs(current_score - best_score) < epsilon:
            return best_idxs, best_score
        
        step_total += 1
        if step_total >= max_steps_total:
            break

    return best_idxs, best_score


def optimize_sample_random_multiple(
    instruction: np.ndarray,
    targets: List[Dict[str, np.ndarray]],
    number: int,
    dim: int,
    epsilon: float = 1e-4,
    max_steps: int = 512,
    retries: int = 10,
    theta: np.ndarray = None
) -> Tuple[List[str], float]:
    if theta is None:
        theta = generateTheta(dim, targets[0]['dataset'].shape[1])

    best_score = float('inf')
    best_idxs: List[str] = []
    for _ in range(retries):
        current_score, current_idxs = optimize_sample_random(
            instruction, targets, number, dim, epsilon=epsilon, max_steps=max_steps, theta=theta)
        if current_score < best_score:
            best_score = current_score
            best_idxs = current_idxs

    return best_idxs, best_score


def optimize_sample_anneal(
    instruction: np.ndarray,
    targets: List[Dict[str, np.ndarray]],
    number: int,
    dim: int,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.99,
    min_temp: float = 1e-4,
    epsilon: float = 5e-4,
    large_step_threshold: int = 100
) -> Tuple[List[str], float]:
    def calculate_score(idxs_selected: List[str], instruction: np.ndarray, targets: List[Dict[str, np.ndarray]], theta: np.ndarray) -> float:
        current_score = 0
        for i in range(len(targets)):
            idxs_selected_current = [
                int(idx.split('.')[-1]) for idx in idxs_selected if idx.startswith(f"{i}.")]
            if not idxs_selected_current:
                continue
            dataset_score = sWasserstein(
                targets[i]['dataset'][idxs_selected_current], instruction, theta)
            instruction_score = sWasserstein(
                targets[i]['instruction'], instruction, theta)
            current_score += len(idxs_selected_current) / \
                number * (6 * dataset_score + instruction_score)
        return current_score

    def adaptive_perturb(current_idxs: List[str], targets: List[Dict[str, np.ndarray]], large_step: bool = False) -> List[str]:
        idxs = [f"{i}.{j}" for i in range(len(targets))
                for j in range(len(targets[i]['dataset']))]
        new_idxs = current_idxs[:]
        replace_prob = 0.5 if large_step else 0.1
        for idx in range(len(new_idxs)):
            if random.random() < replace_prob:
                new_idxs[idx] = random.choice(idxs)
        return new_idxs

    theta = generateTheta(dim, targets[0]['dataset'].shape[1])

    # Generate initial solution with random sampling
    current_idxs, current_score = optimize_sample_random(
        instruction, targets, number, dim, theta=theta, epsilon=epsilon)
    best_idxs = current_idxs
    best_score = current_score
    temp = initial_temp
    no_improvement_count = 0

    # Start Simulated Annealing
    while temp > min_temp:
        large_step = no_improvement_count > large_step_threshold // 2
        new_candidates = [adaptive_perturb(
            current_idxs, targets, large_step) for _ in range(10)]
        new_scores = [calculate_score(
            candidate, instruction, targets, theta) for candidate in new_candidates]

        min_new_score = min(new_scores)
        best_candidate_idx = new_scores.index(min_new_score)

        if min_new_score < best_score:
            # Update best score
            best_score = min_new_score
            best_idxs = new_candidates[best_candidate_idx]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if min_new_score < current_score or random.random() < np.exp((current_score - min_new_score) / temp):
            current_idxs = new_candidates[best_candidate_idx]
            current_score = min_new_score

        temp *= cooling_rate

    return best_idxs, best_score
