import json
import random

from typing import List, Dict, Any, Tuple

random.seed(42)


def split_dataset(
    dataset: Dict[str, List[Dict[str, Any]]],
    dev_size: int,
    train_size: int = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:

    # Calculate the total number of elements in the dataset
    total_elements = sum(len(items) for items in dataset.values())

    # If train_size is None, use all remaining elements after dev_size as train_size
    if train_size is None:
        train_size = total_elements - dev_size

    # Initialize lists to collect sampled items
    train_samples = []
    dev_samples = []

    # Initialize the train and dev sets
    train_set = {}
    dev_set = {}

    # First pass: ensure each key has at least one item in train and dev
    for key, items in dataset.items():
        train_set[key] = []
        dev_set[key] = []

        if len(items) == 1:
            # If there is only one item, add it to train or dev based on the remaining space
            if len(dev_samples) < dev_size:
                dev_set[key].append(items[0])
                dev_samples.append((key, 0))
            else:
                train_set[key].append(items[0])
                train_samples.append((key, 0))
        else:
            # Randomly select one item for dev and one for train
            first_item = random.choice(items)
            dev_set[key].append(first_item)
            dev_samples.append((key, items.index(first_item)))

            remaining_items = [item for item in items if item != first_item]
            second_item = random.choice(remaining_items)
            train_set[key].append(second_item)
            train_samples.append((key, items.index(second_item)))

    # Calculate how many more items are needed for train and dev
    remaining_dev_size = dev_size - len(dev_samples)
    remaining_train_size = train_size - len(train_samples)

    # Create a pool of remaining items
    remaining_items = [(key, index, item) for key, items in dataset.items() for index, item in enumerate(items)
                       if (key, index) not in dev_samples and (key, index) not in train_samples]

    # Shuffle remaining items for random sampling
    random.shuffle(remaining_items)

    # Allocate the remaining items to train and dev sets based on remaining sizes
    for key, index, item in remaining_items:
        if remaining_dev_size > 0:
            dev_set[key].append(item)
            remaining_dev_size -= 1
        elif remaining_train_size > 0:
            train_set[key].append(item)
            remaining_train_size -= 1

        # Break if both sets are filled
        if remaining_dev_size == 0 and remaining_train_size == 0:
            break

    return train_set, dev_set


def dump_dataset(dataset: Dict[str, List[Dict[str, Any]]], path: str, dev_size: int = None, train_size: int = None) -> None:
    if not dev_size:
        dev_size = int(0.1 * sum(len(dataset[key]) for key in dataset))
    train_set, dev_set = split_dataset(dataset, dev_size, train_size)
    print(f"Train set size: {sum(len(train_set[key]) for key in train_set)}")
    print(f"Dev set size: {sum(len(dev_set[key]) for key in dev_set)}")
    with open(f'{path}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)
    with open(f'{path}/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev_set, f, ensure_ascii=False, indent=4)
