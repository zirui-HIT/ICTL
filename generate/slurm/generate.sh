MODEL=Llama3.1-8b
DATASET=Super-NI
ORDER=normal
# for PART in synthesis; do
#     if [ $PART == "ours" ]; then
#         STRATEGY=both
#     elif [ $PART == "direct" ]; then
#         STRATEGY=synthesis
#     else
#         STRATEGY=$PART
#     fi

#     python3 ./generate/generate.py \
#         --strategy $STRATEGY \
#         --model_name_or_path ./model/Llama-3.1-Instruct/8B \
#         --config_file ./config/Llama-3.1-Instruct.json \
#         --test_name_file ./dataset/$DATASET/splits/default/test_tasks.txt \
#         --test_dir ./dataset/$DATASET/tasks \
#         --demo_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART \
#         --output_dir ./generate/result/$MODEL/$DATASET/samping_strategy/$PART \
#         --order $ORDER
# done

for PART in 768 1024 2048; do
    python3 ./generate/generate.py \
        --strategy synthesis \
        --model_name_or_path ./model/Llama-3.1-Instruct/8B \
        --config_file ./config/Llama-3.1-Instruct.json \
        --test_name_file ./dataset/$DATASET/splits/default/test_tasks.txt \
        --test_dir ./dataset/$DATASET/tasks \
        --demo_dir ./transfer/result/$MODEL/$DATASET/target_dataset_scale/$PART \
        --output_dir ./generate/result/$MODEL/$DATASET/target_dataset_scale/$PART \
        --order $ORDER
done
