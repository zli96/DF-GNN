# read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter heads(default=1): " heads
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${dim}" ]; then
	dim=64
fi
if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi
# if [ -z "${format}" ];then
# 	format="csr"
# fi

datasets=(cora arxiv cite pubmed)
formats=(csr hyper)
day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

# for dataset in ${datasets[@]};
# do
# for format in ${formats[@]};
# do
# python -u dgNN/script/test/test_gt_full_graph.py --dim $dim --heads $heads --dataset ${dataset} --data-dir ${data_dir} --format ${format} | tee log/day_${day}/gt_${dataset}_${format}_dim${dim}_h${heads}_${Time}.log
# done
# done

num_neighs=(2 4 8 16 32 64 128)
for format in ${formats[@]}; do
	for num_neigh in ${num_neighs[@]}; do
		python -u dgNN/utils/graph_generate.py --format ${format} --num-neigh ${num_neigh} | tee log/day_${day}/gt_constant_degree_${format}_neigh${num_neigh}_${Time}.log
	done
done
