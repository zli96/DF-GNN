read -p "Enter dataset(default=ogbg-molhiv, ogbg-molpcba): " dataset
read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter heads(default=1): " heads
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${format}" ];then
	format=csr
fi
if [ -z "${dim}" ];then
    dim=96
fi
if [ -z "${heads}" ];then
    heads=1
fi
if [ -z "${data_dir}" ];then
	data_dir="/workspace2/dataset"
fi
if [ -z "${dataset}" ];then
	# dataset="ogbg-molhiv"
    # dataset="Peptides-func"
	# dataset="Peptides-struct"
	dataset="PascalVOC-SP"
	# dataset="COCO-SP"
fi

batch_sizes=(16 32 64 128 256 512 1024 2048)
# batch_sizes=(32)

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop
for bs in ${batch_sizes[@]};
do
    # # run with nolog
    # python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format}

    # # run with log
    python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format}| tee log/day_${day}/gf_${dataset}_${format}_dim${dim}_h${heads}_bs${bs}_${Time}.log
    # python -u dgNN/script/test/test_gf_subgraph.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} | tee log/day_${day}/gf_subgraph_${dataset}_dim${dim}_h${heads}_bs${bs}_${Time}.log
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