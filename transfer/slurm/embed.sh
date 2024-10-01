MODEL=GPT-4o
ORDER=reverse
DATASET=Super-NI

for PART in ours; do
    python3 ./transfer/embed.py \
        --model_name_or_path model/BGE-EN-ICL \
        --config_file config/Llama-3.1-Instruct.json \
        --task_dir ./dataset/$DATASET/tasks \
        --task_file ./dataset/$DATASET/splits/default/test_tasks_tiny.txt \
        --transfer_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/verify \
        --save_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/verify \
        --order $ORDER
done

# python3 ./transfer/embed.py \
#     --model_name_or_path model/BGE-EN-ICL \
#     --config_file config/Llama-3.1-Instruct.json \
#     --task_dir ./dataset/tasks \
#     --task_file ./dataset/splits/default/test_tasks.txt \
#     --transfer_dir ./transfer/result/sampling_strategy/ours/none \
#     --save_dir ./transfer/result/sampling_strategy/ours/none \
#     --order $ORDER

# for NUMBER in 100 10 1 -1; do
#     python3 ./transfer/embed.py \
#         --model_name_or_path model/BGE-EN-ICL \
#         --config_file config/Llama-3.1-Instruct.json \
#         --task_dir ./dataset/$DATASET/tasks \
#         --task_file ./dataset/$DATASET/splits/default/test_tasks.txt \
#         --transfer_dir ./transfer/result/$DATASET/source_domain_similarity/$NUMBER/verify \
#         --save_dir ./transfer/result/$DATASET/source_domain_similarity/$NUMBER/verify \
#         --order $ORDER
# done
