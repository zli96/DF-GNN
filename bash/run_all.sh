#!/bin/bash

datasets=(ogbg-molhiv PATTERN CLUSTER MNIST CIFAR10 Peptides-func Peptides-struct PascalVOC-SP COCO-SP)

formats=(outdegree csr hyper)

batch_sizes=(16 32 64 128 256 512 1024 2048 4096)

data_dir="/workspace2/dataset"

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop
for bs in ${batch_sizes[@]}; do
    for dataset in ${datasets[@]}; do
        config_dir=config/${dataset}_sparse_attention.yaml

        for format in ${formats[@]}; do
            name=gf_${dataset}_${format}_${Time}

            # # run with nolog
            # cuda-gdb -ex r --args  python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs
            # CUDA_LAUNCH_BLOCKING=1 python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs
            # python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs
            
            # # run with log
            python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs | tee -a log/day_${day}/${name}.log

            # # run with nohup
            # echo "nohup python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --dataset ${dataset} --format ${format} --config ${config_dir} --batch-size $bs  >> log/day_${day}/${name}.log 2>&1 &" | bash;
        done
    done
done



datasets=(ogbg-molhiv PATTERN CLUSTER MNIST CIFAR10)

formats=(outdegree csr hyper subgraph)

batch_sizes=(16 32 64 128 256 512 1024 2048 4096)

Time=$(date +%H_%M_%S)

for bs in ${batch_sizes[@]}; do
    for dataset in ${datasets[@]}; do
        config_dir=config/${dataset}_sparse_attention.yaml

        for format in ${formats[@]}; do
            name=gf_subgraph-filter_${dataset}_${format}_${Time}
            # # run with log
            python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs --subgraph-filter | tee -a log/day_${day}/${name}.log
        done
    done
done
