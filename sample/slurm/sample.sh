DATASET=Super-NI
python3 ./sample/run.py \
    --test_file ./dataset/$DATASET/splits/default/test_tasks.txt \
    --task_map ./sample/result/$DATASET/sampling_strategy/large/task_map.json \
    --dump_dir ./sample/result/$DATASET/sampling_strategy/large \
    --embed_dir ./embed/embedding/$DATASET \
    --sample_scale 1024 \
    --order reverse
