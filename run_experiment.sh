

#!/bin/bash
source venv/bin/activate

# Create a bash array using Python's logspace
# readarray -t values < <(python3 -c "import numpy as np; print('\n'.join(map(str, np.unique(np.logspace(np.log10(1), np.log10(2000), num=100, dtype=int)))))")
readarray -t values < <(python3 -c "import numpy as np; print('\n'.join(map(str, np.unique(np.linspace(10, 120, num=80, dtype=int)))))")

NB_RUNS=1
START_FROM=127
END_AT=2000
echo "Generated values:"
echo "${values[@]}"

# Example: loop over each value and run your measurement
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

    
    start=$(date +%s.%N)
    python ingestion_pipeline.py
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "ingestion_pipeline.py took $duration seconds"
    
    # 1. ground_truth_collector.py
    start=$(date +%s.%N)
    python ground_truth_collector.py --dataset RAGBench_whole/merged_id_triplets_no_duplicates.json --runs $NB_RUNS --top-k $val 
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "ground_truth_collector.py took $duration seconds"
    

    # add metadata (reached_chunks and targeted_chunks)
    start=$(date +%s.%N)
    python add_metadata_tag.py
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "add_metadata_tag.py took $duration seconds"



    # experiment rotation
    # start=$(date +%s.%N)
    # python experiment_rot_metadata.py --top-k $val --quiet
    # end=$(date +%s.%N)
    # duration=$(echo "$end - $start" | bc)
    # echo "experiment_rot_metadata.py took $duration seconds"

    # # experiment extra dimension
    # start=$(date +%s.%N)
    # python experiment_extra_dim.py --top-k $val --quiet
    # end=$(date +%s.%N)
    # duration=$(echo "$end - $start" | bc)
    # echo "experiment_extra_dim.py took $duration seconds"

    # experiment extra dimension optimized
    # start=$(date +%s.%N)
    # python opti_extra_dim_experiment.py --top-k $val --quiet
    # end=$(date +%s.%N)
    # duration=$(echo "$end - $start" | bc)
    # echo "experiment_extra_dim.py took $duration seconds"

    ### experiment extra dimension optimized
    # start=$(date +%s.%N)
    # python opti_experiment.py --top-k $val --quiet
    # end=$(date +%s.%N)
    # duration=$(echo "$end - $start" | bc)
    # echo "experiment_extra_dim.py took $duration seconds"

    ### experiment extra dimension optimized raw retrieved
    start=$(date +%s.%N)
    python opti_experiment_raw_retrieve.py --top-k $val --quiet
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "opti_experiment_raw_retrieve.py took $duration seconds"


    python save_ground_truth.py --top-k $val


    rm -rf RAGBench_whole/merged_id_triplets_no_duplicates.json
    rm -rf RAGBench_whole/merged_id_triplets_with_metadata2.json
    rm -rf RAGBench_whole/ground_truth_retrievals.json

    
    echo "--------------------------------------------"

done


# # python ground_truth_collector.py --dataset RAGBench_whole/merged_id_triplets_no_duplicates.json --runs 3 --top-k 10 
# val=20
# echo "=== Running for parameter: $val ==="

# start=$(date +%s.%N)
# python ingestion_pipeline.py
# end=$(date +%s.%N)
# duration=$(echo "$end - $start" | bc)
# echo "ingestion_pipeline.py took $duration seconds"

# # 1. ground_truth_collector.py
# start=$(date +%s.%N)
# python ground_truth_collector.py --dataset RAGBench_whole/merged_id_triplets_no_duplicates.json --runs $NB_RUNS --top-k $val 
# end=$(date +%s.%N)
# duration=$(echo "$end - $start" | bc)
# echo "ground_truth_collector.py took $duration seconds"

# start=$(date +%s.%N)
# python add_metadata_tag.py
# end=$(date +%s.%N)
# duration=$(echo "$end - $start" | bc)
# echo "add_metadata_tag.py took $duration seconds"

# # rm -rf RAGBench_whole/merged_id_triplets_no_duplicates.json
# # rm -rf RAGBench_whole/merged_id_triplets_with_metadata2.json
# # rm -rf RAGBench_whole/ground_truth_retrievals.json
