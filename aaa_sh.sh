#!/bin/bash

SESSION_NAME="mysession"
LAUNCH_EMBEDDING_SERVER_SCRIPT="$(pwd)/start_embedding_server.sh"

NB_RUNS=1 # for ground_truth_collector.py
START_FROM=1
END_AT=2000
NUMBER_QUERIES=30
NAME_LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).log"
NEED_RECREATE_VECTOR_STORE="False"
NB_USERS=20



source venv/bin/activate

if [[ ! -f "$LAUNCH_EMBEDDING_SERVER_SCRIPT" ]]; then
    echo "Error: Target script '$LAUNCH_EMBEDDING_SERVER_SCRIPT' not found."
    exit 1
fi

if [[ ! -x "$LAUNCH_EMBEDDING_SERVER_SCRIPT" ]]; then
    echo "Error: Target script '$LAUNCH_EMBEDDING_SERVER_SCRIPT' is not executable."
    echo "Run: chmod +x \"$LAUNCH_EMBEDDING_SERVER_SCRIPT\""
    exit 1
fi

# Start server and log output
screen -dmS "$SESSION_NAME" bash -c "$LAUNCH_EMBEDDING_SERVER_SCRIPT # >> $NAME_LOG_FILE 2>&1"

if [[ $? -eq 0 ]]; then
    echo "Started '$LAUNCH_EMBEDDING_SERVER_SCRIPT' in detached screen session: $SESSION_NAME"
    echo "Logging output to: $NAME_LOG_FILE"
    echo "To attach: screen -r $SESSION_NAME"
else
    echo "Failed to start screen session."
    exit 1
fi

sleep 20 # wait for the server to start

# Generate, log, and assign python output
mapfile -t values < <(python3 -c "import numpy as np; print('\n'.join(map(str, np.unique(np.linspace(10, 120, num=80, dtype=int)))))" )


if [[ "$NEED_RECREATE_VECTOR_STORE" == "True" ]]; then
    # create the dataset
    echo "get RAGBench.py is running..."
    python RAGBench_whole/getRAGBench.py --max_length $NUMBER_QUERIES #>> "$NAME_LOG_FILE" 2>&1

    # create the vector database

    start=$(date +%s.%N)
    python ingestion_pipeline.py --create_vector_store True >> "$NAME_LOG_FILE" 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "ingestion_pipeline.py took $duration seconds" | tee -a "$NAME_LOG_FILE"
fi



for val in "${values[@]}"
do
    # Skip values until we hit START_FROM
    if (( val < START_FROM )); then
        continue
    fi
    # Stop the loop if we exceed END_AT
    if (( val > END_AT )); then
        break
    fi

    echo "=== Measuring for parameter: $val ==="

    python nuclear_cleanup.py

    
    start=$(date +%s.%N)
    python ingestion_pipeline.py --create_vector_store False >> "$NAME_LOG_FILE" 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "ingestion_pipeline.py took $duration seconds" | tee -a "$NAME_LOG_FILE"
    
    # 1. ground_truth_collector.py
    start=$(date +%s.%N) 
    python ground_truth_collector.py --dataset RAGBench_whole/merged_id_triplets_no_duplicates.json --runs $NB_RUNS --top-k $val >> "$NAME_LOG_FILE" 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "ground_truth_collector.py took $duration seconds" | tee -a "$NAME_LOG_FILE"
    

    # add metadata (reached_chunks and targeted_chunks)
    start=$(date +%s.%N)
    python add_metadata_tag.py --nb-users $NB_USERS >> "$NAME_LOG_FILE" 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "add_metadata_tag.py took $duration seconds" | tee -a "$NAME_LOG_FILE"

    ### experiment extra dimension optimized raw retrieved
    start=$(date +%s.%N)
    python opti_experiment_raw_retrieve_copy.py --top-k $val --workers 64 >> "$NAME_LOG_FILE" 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "opti_experiment_raw_retrieve_copy.py took $duration seconds" | tee -a "$NAME_LOG_FILE"


    python save_ground_truth.py --top-k $val


    # rm -rf RAGBench_whole/merged_id_triplets_no_duplicates.json
    # rm -rf RAGBench_whole/merged_id_triplets_with_metadata2.json
    # rm -rf RAGBench_whole/ground_truth_retrievals.json
    # killall screen

    
    echo "--------------------------------------------"

done

