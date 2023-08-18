day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

dataset=PATTERN
# dataset=CLUSTER
# dataset=CIFAR10
# dataset=MNIST

dim=32
heads=1
bs=2048

# export BLOCK_SIZE=8
export BLOCK_SIZE=32

python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} | tee log/day_${day}/gf_subgraph_${dataset}_blocksize${BLOCK_SIZE}_dim${dim}_h${heads}_bs${bs}_${Time}.log
# python -u dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} 
