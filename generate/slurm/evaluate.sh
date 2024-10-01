MODEL=GPT-4o
DATASET=Super-NI
for PART in ours; do
    DATA_DIR=$DATASET/sampling_strategy/$PART
    echo "Evaluating $DATA_DIR"
    python3 ./generate/merge.py \
        --load_dir ./generate/result/$MODEL/$DATA_DIR \
        --data_file ./dataset/$DATASET/splits/default/test_tasks_tiny.txt
    python3 ./generate/evaluate.py \
        --prediction_file ./generate/result/$MODEL/$DATA_DIR/prediction.jsonl \
        --reference_file ./dataset/$DATASET/reference/default/test_references_tiny.jsonl \
        --output_file ./generate/result/$MODEL/$DATA_DIR/prediction.eval.json
    python3 ./generate/analyze.py \
        --pred_file ./generate/result/$MODEL/$DATA_DIR/prediction.eval.json \
        --category_file ./dataset/$DATASET/category.json \
        --task_file ./dataset/$DATASET/splits/default/test_tasks_tiny.txt
done
