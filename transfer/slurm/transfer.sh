MODEL=Llama3.1-8b
DATASET=Super-NI
ORDER=reverse

for PART in large; do
    python3 transfer/transfer.py \
        --model_name_or_path model/Llama-3.1-Instruct/8B \
        --config_file config/Llama-3.1-Instruct.json \
        --task_dir ./dataset/$DATASET/tasks \
        --task_file ./dataset/$DATASET/splits/default/test_tasks.txt \
        --sample_dir ./sample/result/$DATASET/sampling_strategy/$PART \
        --dump_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/none \
        --order $ORDER
done

# for NUMBER in -1 1 10 100; do
#     python3 transfer/transfer.py \
#         --model_name_or_path model/Llama-3.1-Instruct/8B \
#         --config_file config/Llama-3.1-Instruct.json \
#         --task_dir ./dataset/tasks \
#         --task_file ./dataset/splits/default/test_tasks.txt \
#         --sample_dir ./sample/result/source_domain_similarity/$NUMBER \
#         --dump_dir ./transfer/result/source_domain_similarity/$NUMBER/none \
#         --order $ORDER
# done
