data_dir="/workspace2/dataset"



datasets=(Peptides-func Peptides-struct PascalVOC-SP COCO-SP PascalVOC-SP ogbg-molhiv PATTERN CLUSTER MNIST CIFAR10)
# datasets=(ogbg-molhiv)

for dataset in ${datasets[@]}; do
    config_dir=config/${dataset}_sparse_attention.yaml
    python -u dgNN/utils/graph_statistics.py --data-dir ${data_dir} --config ${config_dir} --dataset ${dataset}
done
