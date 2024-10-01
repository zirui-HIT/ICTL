import os
import sys
import json
import random
import argparse

from typing import Dict, Any, List

random.seed(42)
sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPT: Dict[str, str] = {
    "zero": """
{instruction}
Think it step by step, and present your answer with "Reason:\n<Explanation of the answer>\nAnswer:\n<Your answer>\n".
{user_input}
""".strip(),
    "label": """
{instruction}
Here are some demonstrations of the task:

---

{demonstration}

---

Based on the above demonstrations, please generate a response to the following question.
Your output format should be as follows:
Reason:
<Explanation of the answer>
Answer:
<Your answer>
Think it step by step.

Input:
{user_input}
""".strip()
}

PROMPT_EXAMPLE = """
Input:
{input}
Reason:
{reason}
Answer:
{answer}
""".strip()


def fix_answer(answer: str) -> str:
    try:
        answer = answer.split('Answer:')[-1].strip()
    except:
        answer = answer.strip()
    answer = answer.strip('.:').strip()
    return answer


def pack_demo(demo_used: List[List[Dict[str, Any]]], data: List[Dict[str, Any]]) -> List[str]:
    prompts = []
    for d, demos in zip(data, demo_used):
        prompts_demo = [PROMPT_EXAMPLE.format(
            input=x['question'],
            reason=x['rationale'],
            answer=x['answer']
        ) for x in demos]
        prompts.append(PROMPT['label'].format(
            instruction=dataset['Definition'][0],
            demonstration='\n\n---\n\n'.join(prompts_demo),
            user_input=d['question']
        ))
    return prompts


if __name__ == '__main__':
    from utils.selector import select_multiple
    from utils.generator import LlamaChatGenerator, pack_answer

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='human',
                        choices=['zero', 'human', 'synthesis', 'both'])
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--test_name_file', type=str)
    parser.add_argument('--demo_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--order', type=str, default='normal',
                        choices=['normal', 'reverse', 'random'])
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--task_scale', type=int)
    args = parser.parse_args()

    generator = LlamaChatGenerator(args.model_name_or_path)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(args.test_name_file, 'r', encoding='utf-8') as f:
        data_file_names = [x.strip() for x in f if x.strip()]
    if args.task_scale:
        data_file_names = data_file_names[:args.task_scale]
    data_file_names = sorted(
        data_file_names, reverse=(args.order == 'reverse'))
    if args.order == 'random':
        random.shuffle(data_file_names)
    for file_name in data_file_names:
        dump_file = os.path.join(args.output_dir, file_name + '.json')
        if os.path.exists(dump_file):
            print(f"Skip {file_name}")
            continue

        with open(os.path.join(args.test_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            data: List[Dict[str, Any]] = [{
                "id": d['id'],
                "question": d['input'],
                "answer": d['output']
            } for d in dataset['Instances']][:100]  # 100 examples randomly sampled by Natural-Instruction
            if args.use_all_data:
                dataset['Positive Examples'] += [{
                    "input": d['input'],
                    "output": d['output'][0],
                    "explanation": ""
                } for d in data[100:]]

        if args.strategy == "zero":
            prompts = [PROMPT['zero'].format(
                instruction=dataset['Definition'][0],
                user_input=d['question']
            ) for d in data]
        elif args.strategy == "human":
            demo: List[Dict[str, Any]] = [{
                "question": d['input'],
                "answer": d['output'],
                "rationale": d['explanation']
            } for d in dataset['Positive Examples']]
            demo_used = select_multiple(data, demo, demonstration_number=1)
            prompts = pack_demo(demo_used, data)
        else:
            with open(os.path.join(args.demo_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
                demo_synthesis = [{
                    "question": d['instance']['input'],
                    "answer": d['instance']['output'],
                    "rationale": d['instance']['explanation']
                } for d in json.load(f)]
            demo_human = [{
                "question": d['input'],
                "answer": d['output'],
                "rationale": d['explanation']
            } for d in dataset['Positive Examples']]
            if args.strategy == "synthesis":
                demo_used = select_multiple(data, demo_synthesis)
            elif args.strategy == "both":
                demo_used = select_multiple(data, demo_synthesis + demo_human)
            demo_used = [s[:3] for s in demo_used]
            prompts = pack_demo(demo_used, data)
        print(prompts[-1])

        responses = generator.generate(prompts, config)
        results = []
        for d, r in zip(data, responses):
            ans = fix_answer(r[0][0])
            results.append({
                'id': d['id'],
                'answer': d['answer'],
                'prediction': ans,
                'prediction_origin': r[0][0].strip().split('\n')
            })
        with open(dump_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
