if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
log_dir=log_new/day_${day}/full_fg_super
mkdir -p ${log_dir}

set -e
python setup.py develop

convs=(gt agnn)
# datasets=(ppa reddit protein)
# formats=(pyg csr cugraph softmax_gm)
# dims=(128)

# for conv in ${convs[@]}; do
# 	for dim in ${dims[@]}; do
# 		for dataset in ${datasets[@]}; do
# 			for format in ${formats[@]}; do
# 				name=${conv}_${dataset}_${format}_dim${dim}_${Time}
# 				python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv} --store-result 2>&1 |
# 					tee log/day_${day}/${name}.log
# 			done
# 		done
# 	done
# done

convs=(gat)
datasets=(ppa reddit protein)
formats=(csr softmax_gm tiling)
formats=(pyg)
dims=(128)

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for format in ${formats[@]}; do
				name=${conv}_${dataset}_${format}_dim${dim}_${Time}
				python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv} --store-result 2>&1 |
					tee ${log_dir}/${name}.log
			done
		done
	done
done

# # convs=(gt agnn)
# convs=(gat)
# datasets=(ppa reddit protein)
# format=all_fg_super
# dims=(128)

# for conv in ${convs[@]}; do
# 	for dim in ${dims[@]}; do
# 		for dataset in ${datasets[@]}; do
# 			name=${conv}_${dataset}_${format}_dim${dim}_${Time}
# 			python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv} --store-result 2>&1 | tee $log_dir/${name}.log
# 		done
# 	done
# done
