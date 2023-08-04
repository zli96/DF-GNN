set -e
python setup.py develop
python dgNN/script/test/test_gf_subgraph.py --batch-size 512 --dataset CLUSTER --dim 32
# python dgNN/script/test/test_gf.py --batch-size 512 --dataset CLUSTER --dim 32 --heads 1

