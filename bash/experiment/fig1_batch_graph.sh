if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/DFGNN/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
log_dir=log_new/day_${day}/fig_bg
mkdir -p ${log_dir}

convs=(gt agnn gat)
format=all
dims=(128)
bs=1024
datasets=(COCO-SP PascalVOC-SP MNIST CIFAR10 CLUSTER PATTERN)

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			name=${conv}_${dataset}_${format}_dim${dim}_${Time}
			#store result
			cmd="python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result"

			# #not store result
			# cmd="python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv}"
			echo $cmd | tee ${log_dir}/${name}.log
			$cmd 2>&1 | tee -a ${log_dir}/${name}.log
		done
	done
done

python DFGNN/utils/plot_fig1.py > log_new/fig1.log