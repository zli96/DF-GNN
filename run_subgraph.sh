read -p "Enter dataset(default=ogbg-molhiv, ogbg-molpcba): " dataset
read -p "Enter dim(default=64): " dim

if [ -z "${dim}" ];then
    dim=64
fi
if [ -z "${dataset}" ];then
	dataset="ogbg-molhiv"
	# dataset="PATTERN"
fi


day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}


# dataset=PATTERN
# dataset=CLUSTER
# dataset=CIFAR10
# dataset=MNIST


export BLOCK_SIZE=32
# export BLOCK_SIZE=32
# batch_sizes=(32 64 128 256 512 1024 2048 4096)
batch_sizes=(2048)

set -e
python setup.py develop
for bs in ${batch_sizes[@]};
do
    python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} | tee log/day_${day}/gf_subgraph_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_h${heads}_bs${bs}_${Time}.log
    # CUDA_LAUNCH_BLOCKING=1 python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} 
    # python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} 

done