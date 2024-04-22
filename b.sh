#!/bin/bash

# Array of batch sizes to test
# batch_sizes=(1 8 64 256)
batch_sizes=(1 2 4 8 16 32)
# batch_sizes=(512 1024)

# Array of thread pool sizes to test
thread_pools=(8)

# Array of models to test
models=("facebook/opt-13b")
input_lens=(64 128 256 512 1024 2048)

# Loop through each batch size
for input_len in "${input_lens[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        # Loop through each thread pool size
        for tp in "${thread_pools[@]}"; do
            # Loop through each model
            for model in "${models[@]}"; do
                # Run with enforce-eager
                # echo "\n===================\nRunning with --enforce-eager --batch-size $batch_size -tp $tp --model $model\n===================\n"
                # python a.py --enforce-eager --batch-size $batch_size -tp $tp --model $model --input-len $input_len
                # sleep 5

                # Run without enforce-eager
                echo "\n===================\nRunning without --enforce-eager --batch-size $batch_size -tp $tp --model $model\n===================\n"
                python a.py --batch-size $batch_size -tp $tp --model $model --input-len $input_len
                sleep 5
            done
        done
    done
done
