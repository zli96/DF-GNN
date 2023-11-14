conv=gt

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

datasets=(PATTERN)
formats=(hyper)
batch_sizes=(4096)
dims=(64)
echo test mode !!!!!!!!!!!!

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for format in ${formats[@]}; do
			for bs in ${batch_sizes[@]}; do
				python -u dgNN/script/test/test_gt_train.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv}

			done
		done
	done
done

# # full-graph
# datasets=("cora" "cite" "pubmed")
# for dataset in ${datasets[@]};
# do
#     mkdir log/day_${day}/${dataset}
#     # python -u dgNN/script/test/test_gt_full_graph.py --dim $dim --heads $heads --dataset $dataset  |tee log/day_${day}/${dataset}/weight_ver_${dim}_${heads}_${Time}.log
# 	python dgNN/script/figure/plot_full_graph.py --dataset $dataset > log/${dataset}_neigh_dist.log
# done
