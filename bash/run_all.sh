#!/bin/bash

datasets=(ogbg-molhiv PATTERN CLUSTER Peptides-func Peptides-struct PascalVOC-SP COCO-SP)

formats=(csr hyper)

batch_sizes=(16 32 64 128 256 512 1024 2048 4096)

data_dir="/workspace2/dataset"

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop
for dataset in ${datasets[@]}; do
    config_dir=config/${dataset}_sparse_attention.yaml

    for format in ${formats[@]}; do
        name=gf_tiling_${dataset}_${format}_${Time}
        for bs in ${batch_sizes[@]}; do

            # # run with nolog
            # python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs

            # # run with log
            python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs | tee -a log/day_${day}/${name}.log

            # # run with nohup
            # echo "nohup python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --dataset ${dataset} --format ${format} --config ${config_dir} --batch-size $bs  >> log/day_${day}/${name}.log 2>&1 &" | bash;
        done
    done
done
