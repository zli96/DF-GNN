# read -p "Enter dim(default=64): " dim
# read -p "Enter heads(default=8): " heads
# # read -p "Enter batch size(default=256): " bs
# read -p "Enter comment: " comment
# read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${dim}" ];then
    dim=512
fi
if [ -z "${heads}" ];then
    heads=1
fi
if [ -z "${comment}" ];then
	comment=normal
fi
if [ -z "${data_dir}" ];then
	data_dir="/workspace2/dataset"
fi


# batch_sizes=(1024 2048 4096)
batch_sizes=(32 64 128 256 512 1024 2048 4096)



day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
python setup.py develop
mkdir log/day_${day}
for bs in ${batch_sizes[@]};
do
    python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} | tee log/day_${day}/gf_dim${dim}_h${heads}_bs${bs}_${comment}_${Time}.log
    # python -u dgNN/script/test/test_gf_ell.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} | tee log/day_${day}/gf_ell_dim${dim}_h${heads}_bs${bs}_${comment}_${Time}.log
    
    # echo "nohup python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs  > log/day_${day}/gf_${dim}_${heads}_${bs}_${comment}_${Time}.log 2>&1 &" | bash;
    # echo "nohup python -u dgNN/script/test/test_gf_ell.py --dim $dim --heads $heads --batch-size $bs  > log/day_${day}/gf_ell_${dim}_${heads}_${bs}_${comment}_${Time}.log 2>&1 &" | bash;

done


# # full-graph
# datasets=("cora" "cite" "pubmed")
# for dataset in ${datasets[@]};
# do
#     mkdir log/day_${day}/${dataset}
#     # python -u dgNN/script/test/test_gf_full_graph.py --dim $dim --heads $heads --dataset $dataset  |tee log/day_${day}/${dataset}/weight_ver_${dim}_${heads}_${Time}.log
# 	python dgNN/script/figure/plot_full_graph.py --dataset $dataset > log/${dataset}_neigh_dist.log
# done