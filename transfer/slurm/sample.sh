MODEL=GPT-4o
DATASET=Super-NI
for PART in ours; do
    python3 ./transfer/optimize.py \
        --test_file ./dataset/$DATASET/splits/default/test_tasks_tiny.txt \
        --transfer_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART/verify \
        --dump_dir ./transfer/result/$MODEL/$DATASET/sampling_strategy/$PART \
        --source_embed_dir ./sample/result/$DATASET/sampling_strategy/$PART \
        --sample_scale 512
done
