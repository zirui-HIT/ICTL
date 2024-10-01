DATASET=Super-NI
for SAMPLE in large; do
    python3 sample/embed.py \
        --test_tasks ./dataset/$DATASET/splits/default/test_tasks.txt \
        --sample_dir ./sample/result/$DATASET/sampling_strategy/$SAMPLE \
        --embed_dir ./embed/embedding/$DATASET
done
