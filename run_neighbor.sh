read -p "Enter dim(default=64): " dim
read -p "Enter block size(default=8): " blk

if [ -z "${dim}" ];then
    dim=64
fi
if [ -z "${blk}" ];then
    blk=8
fi
dims=(32 64 128 256)
datasets=(ogbg-molhiv PATTERN CLUSTER MNIST CIFAR10)
# datasets=(PATTERN CLUSTER MNIST CIFAR10)

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop
for dataset in ${datasets[@]};
do
    python -u dgNN/utils/neighbor_overlap.py --dim $dim --dataset ${dataset} --blocksize ${blk}| tee log/day_${day}/neigh_overlap_${dataset}_dim${dim}_blk${blk}_${Time}.log
done