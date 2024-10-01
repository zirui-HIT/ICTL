import os
import sys
import json
import random
import argparse

from typing import List, Dict, Any

random.seed(42)
sys.path.append('.')


PROMPT = """
Convert an example from Task A into an example for Task B, ensuring that both examples are consistent in terms of domain and knowledge. A sample for Task A is provided below. Please create a corresponding example for Task B, while maintaining the same domain and knowledge context.
The definition of Task A: {task_a_definition} 
The definition of Task B: {task_b_definition}

---

For example, given the following example for Task A:
Input:
{task_a_question_demo}
Reason:
{task_a_rationale_demo}
Answer:
{task_a_answer_demo}

The corresponding example for Task B could be:
Input:
{task_b_question_demo}
Reason:
{task_b_rationale_demo}
Answer:
{task_b_answer_demo}

---

Based on the above example, please transfer the following example from Task A to Task B:
Input:
{task_a_question}
Answer:
{task_a_answer}

Your output format should be as follows:
Input:
<Converted input of Task B>
Reason:
<Explanation of the converted>
Answer:
<Converted answer of Task B>
""".strip()


def unpack(input_string: str) -> Dict[str, str]:
    lines = input_string.strip().split('\n')
    text: Dict[str, str] = {
        'input': '',
        'output': '',
        'explanation': ''
    }
    current_part = None
    for line in lines:
        if line.startswith("Answer:"):
            current_part = 'output'
            text[current_part] = ""
        elif line.startswith("Reason:"):
            current_part = 'explanation'
            text[current_part] = ""
        elif line.startswith("Input:"):
            current_part = 'input'
            text[current_part] = ""
        elif not line.strip():
            current_part = None
        elif current_part is not None:
            text[current_part] += line + '\n'
    for key in text:
        text[key] = text[key].strip()
    text['output'] = text['output'].strip('.').strip()
    return text


if __name__ == '__main__':
    from utils.generator import LlamaChatGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--task_dir', type=str)
    parser.add_argument('--task_file', type=str)
    parser.add_argument('--sample_dir', type=str)
    parser.add_argument('--dump_dir', type=str)
    parser.add_argument(
        '--order', type=str, choices=['random', 'reverse', 'normal'], default='normal')
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    generator = LlamaChatGenerator(args.model_name_or_path)

    with open(args.task_file, 'r', encoding='utf-8') as f:
        data_file_names = f.read().split('\n')
    data_file_names = sorted(
        data_file_names, reverse=(args.order == 'reverse'))
    if args.order == 'random':
        random.shuffle(data_file_names)
    for file_name in data_file_names:
        sample_file = os.path.join(args.sample_dir, file_name + '.json')
        dump_file = os.path.join(args.dump_dir, file_name + '.json')
        if not os.path.exists(sample_file):
            print(f"File {sample_file} not found.")
            continue
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists.")
            continue

        # Load the dataset and demonstrations
        with open(os.path.join(args.task_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        with open(sample_file, 'r', encoding='utf-8') as f:
            demonstrations_idx: Dict[str, List[str]] = json.load(f)
        demonstrations: List[Dict[str, Any]] = []
        for demonstration_task_name, instances_idx in demonstrations_idx.items():
            with open(os.path.join(args.task_dir, demonstration_task_name + '.json'), 'r', encoding='utf-8') as f:
                demonstration_task = json.load(f)
            demonstrations.extend([{
                'definition': demonstration_task['Definition'],
                'example': demonstration_task['Positive Examples'],
                'idx': instance['id'],
                'question': instance['input'],
                'answer': instance['output']
            } for instance in demonstration_task['Instances'] if instance['id'] in instances_idx])

        # Generate prompts
        prompts: List[str] = []
        source_demonstration: List[str] = []
        for demo in demonstrations:
            for source_example in demo['example']:
                for target_example in dataset['Positive Examples']:
                    for demo_answer in demo['answer']:
                        prompts.append(PROMPT.format(
                            task_a_definition=demo['definition'][0],
                            task_b_definition=dataset['Definition'][0],
                            task_a_question_demo=source_example['input'],
                            task_a_rationale_demo=source_example['explanation'],
                            task_a_answer_demo=source_example['output'],
                            task_b_question_demo=target_example['input'],
                            task_b_rationale_demo=target_example['explanation'],
                            task_b_answer_demo=target_example['output'],
                            task_a_question=demo['question'],
                            task_a_answer=demo_answer
                        ))
                    source_demonstration.append(demo['idx'])
        print(prompts[-1])

        # Generate converted examples
        converted_examples = generator.generate(prompts, config)
        results = []
        for source_idx, example in zip(source_demonstration, converted_examples):
            results.extend([{
                'source_idx': source_idx,
                'instance': unpack(pred[0]),
                'original': pred[0].strip().split('\n')
            } for pred in example])
        with open(dump_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
