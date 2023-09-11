#!/bin/bash

read -p "Enter dim(default=64): " dim

if [ -z "${dim}" ]; then
    dim=32
fi

# if [ -z "${dataset}" ];then
# 	# dataset="ogbg-molhiv"
# 	dataset="CLUSTER"
# 	# dataset="PATTERN"
# fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}
mkdir log/ncu/day_${day}

batch_sizes=(32 64 128 256 512 1024 2048 4096)
# batch_sizes=(32 2048)
# batch_sizes=(32)


# datasets=(PATTERN CLUSTER MNIST CIFAR10)
export BLOCK_SIZE=32
datasets=(PATTERN)


set -e
python setup.py develop

for dataset in ${datasets[@]}; do
    for bs in ${batch_sizes[@]}; do
        # # run without log
        # python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim}

        # # run with log
        name=gf_subgraph_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_bs${bs}_${Time}
        python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} | tee log/day_${day}/${name}.log

        # # debug
        # CUDA_LAUNCH_BLOCKING=1 python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim}

        # # run ncu profile
        # name=ncu_gf_subgraph_no_r2r3_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_bs${bs}_${Time}
        # ncu --set full --import-source yes -c 10 -o log/ncu/day_${day}/${name} -k "fused_forward_kernel_subgraph" python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} --profile  > log/ncu/day_${day}/${name}.log 2>&1
    done
    wait
done
