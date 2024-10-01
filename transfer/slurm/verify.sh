MODEL=Llama3.1-8b
DATASET=Super-NI
ORDER=random

for PART in large; do
    python3 transfer/verify.py \
        --config_file ./config/Llama-3.1-Instruct.json \
        --model_name_or_path ./model/Llama-3.1-Instruct/8B \
        --task_dir ./dataset/$DATASET/tasks \
        --task_file ./dataset/$DATASET/splits/default/test_tasks.txt \
        --transfer_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/none \
        --dump_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/verify \
        --order $ORDER
done

# for NUMBER in 100 10 1 -1; do
#     python3 transfer/verify.py \
#         --config_file ./config/Llama-3.1-Instruct.json \
#         --model_name_or_path ./model/Llama-3.1-Instruct/8B \
#         --task_dir ./dataset/tasks \
#         --task_file ./dataset/splits/default/test_tasks.txt \
#         --transfer_dir ./transfer/result/source_domain_similarity/$NUMBER/none \
#         --dump_dir ./transfer/result/source_domain_similarity/$NUMBER/verify \
#         --order $ORDER
# done
