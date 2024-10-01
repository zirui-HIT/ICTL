import os
import sys
import json
import random
import argparse

random.seed(42)
sys.path.append('.')

PROMPT = """
Given a task description, several examples, and a pre-synthesized example, evaluate whether the pre-synthesized example matches the format and functionality of the provided examples and aligns with the task description. Based on the evaluation, determine whether the pre-synthesized example is "Qualified"
You should check the pre-synthesized example based on the following criteria:
1. Format Consistency: Does the pre-synthesized example follow the format of the provided examples?
2. Task Fulfillment: Does the pre-synthesized example fulfill the requirements of the task description?
3. Functional Accuracy: Are the input and output in the pre-synthesized example consistent with those in the provided examples?
If the pre-synthesized example meets all the criteria above, return: "Qualified."
If the pre-synthesized example fails to meet any of the criteria, return: "Unqualified."
Think it step by step.

Task Description:
{definition}

Examples:

{examples}

Pre-synthesized Example:

{example_synthesized}
""".strip()

PROMPT_EXAMPLE = """
Input:
{input}
Reason:
{reason}
Answer:
{answer}
""".strip()

if __name__ == '__main__':
    from utils.generator import LlamaChatGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--task_dir', type=str)
    parser.add_argument('--task_file', type=str)
    parser.add_argument('--transfer_dir', type=str)
    parser.add_argument('--dump_dir', type=str)
    parser.add_argument('--order', type=str, choices=['random', 'reverse', 'normal'], default='normal')
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
        load_file = os.path.join(args.transfer_dir, file_name + '.json')
        dump_file = os.path.join(args.dump_dir, file_name + '.json')
        if os.path.exists(dump_file):
            print(f'{dump_file} exists, skip')
            continue
        if not os.path.exists(load_file):
            print(f'{load_file} not exists, skip')
            continue

        with open(os.path.join(args.task_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
            dataset_origin = json.load(f)
        with open(os.path.join(args.transfer_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        prompt_origin = '\n\n'.join([PROMPT_EXAMPLE.format(
            input=d['input'],
            reason=d['explanation'],
            answer=d['output']
        ) for d in dataset_origin['Positive Examples']])
        prompts = [PROMPT.format(
            definition=dataset_origin['Definition'][0],
            examples=prompt_origin,
            example_synthesized=PROMPT_EXAMPLE.format(
                input=d['instance']['input'],
                reason=d['instance']['explanation'],
                answer=d['instance']['output']
            )
        ) for d in dataset if d['instance']]
        print(prompts[-1])

        predictions = generator.generate(prompts, config)
        results = []
        for d, p in zip(dataset, predictions):
            # d['verification'] = {
            #     "qualified": not ('Unqualified' in p[0][0]),
            #     "origin": p[0][0].strip().split('\n')
            # }
            if not ('Unqualified' in p[0][0]):
                results.append(d)
        with open(os.path.join(args.dump_dir, file_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
