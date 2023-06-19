# read -p "Enter dim(default=64): " dim
# read -p "Enter heads(default=8): " heads
# # read -p "Enter batch size(default=256): " bs
# read -p "Enter comment: " comment

if [ -z "${dim}" ];then
    dim=64
fi
if [ -z "${heads}" ];then
    heads=1
fi
if [ -z "${comment}" ];then
    comment=shuffle
fi

# batch_sizes=(256 128 64 32)
batch_sizes=(512 1024 2048 4096)



day=$(date +%d)
Time=$(date +%H_%M_%S)
python setup.py develop
mkdir log/day_${day}
# for bs in ${batch_sizes[@]};
# do
#     python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs  | tee log/day_${day}/weight_ver_${dim}_${heads}_${bs}_${comment}_${Time}.log
#     # nohup python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs  > log/day_${day}/weight_ver_${dim}_${heads}_${bs}_${comment}_${Time}.log 2>&1 &
# done


# full-graph
datasets=("cora" "cite" "pubmed")
for dataset in ${datasets[@]};
do
    mkdir log/day_${day}/${dataset}
    # python -u dgNN/script/test/test_gf_full_graph.py --dim $dim --heads $heads --dataset $dataset  |tee log/day_${day}/${dataset}/weight_ver_${dim}_${heads}_${Time}.log
	python dgNN/script/figure/plot_full_graph.py --dataset $dataset > log/${dataset}_neigh_dist.log
done