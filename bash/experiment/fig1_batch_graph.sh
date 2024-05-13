if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
log_dir=log_new
mkdir ${log_dir}/day_${day}

set -e
python setup.py develop

# convs=(gt agnn gat)
# datasets=(COCO-SP PascalVOC-SP MNIST CIFAR10 CLUSTER PATTERN)
# formats=(csr cugraph softmax hyper)
# dims=(128)
# bs=1024

# for conv in ${convs[@]}; do
# 	for dim in ${dims[@]}; do
# 		for dataset in ${datasets[@]}; do
# 			for format in ${formats[@]}; do
# 				name=${conv}_${dataset}_${format}_dim${dim}_${Time}
# 				python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result 2>&1 | tee -a log/day_${day}/${name}.log
# 			done
# 		done
# 	done
# done

convs=(gt agnn gat)
datasets=(CLUSTER PATTERN COCO-SP PascalVOC-SP MNIST CIFAR10)
format=all
dims=(128)
bs=1024

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			name=${conv}_${dataset}_${format}_dim${dim}_${Time}
			cmd="python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result"
			echo $cmd | tee ${log_dir}/day_${day}/${name}.log
			$cmd 2>&1 | tee -a ${log_dir}/day_${day}/${name}.log
		done
	done
done
