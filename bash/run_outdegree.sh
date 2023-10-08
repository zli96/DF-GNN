#!/bin/bash

read -p "Enter dim(default=64): " dim

if [ -z "${dim}" ]; then
    dim=64
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}
mkdir log/ncu/day_${day}

## specify batch-size
# batch_sizes=(32 64 128 256 512 1024 2048 4096)
# batch_sizes=(2048)
batch_sizes=(32)

## specify dataset
# datasets=(PATTERN CLUSTER MNIST CIFAR10)
datasets=(ogbg-molhiv)

BLOCK_SIZE=$((1024 / ${dim}))

set -e
python setup.py develop

for dataset in ${datasets[@]}; do
    for bs in ${batch_sizes[@]}; do
        # # run without log
        python -u dgNN/script/test/test_gf_outdegree.py --batch-size ${bs} --dataset ${dataset} --dim ${dim}

        # # run with log
        # name=gf_outdegree_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_bs${bs}_${Time}
        # python -u dgNN/script/test/test_gf_outdegree.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} | tee log/day_${day}/${name}.log

        # # debug
        # CUDA_LAUNCH_BLOCKING=1 python -u dgNN/script/test/test_gf_outdegree.py --batch-size ${bs} --dataset ${dataset} --dim ${dim}

        # # run ncu profile
        # name=ncu_gf_outdegree_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_bs${bs}_${Time}
        # ncu --set full --import-source yes -c 10 -o log/ncu/day_${day}/${name} -k "fused_forward_kernel_outdegree_mul32" python -u dgNN/script/test/test_gf_outdegree.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} --profile  > log/ncu/day_${day}/${name}.log 2>&1
    done
    wait
done
