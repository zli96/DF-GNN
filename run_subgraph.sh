day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
# python setup.py developn

# dataset=CLUSTER
dataset=CIFAR10

dim=32
heads=1
bs=16

python dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} | tee log/day_${day}/gf_subgraph_${dataset}_dim${dim}_h${heads}_bs${bs}_${Time}.log
# python dgNN/script/test/test_gf_subgraph.py --batch-size ${bs} --dataset ${dataset} --dim ${dim} 
