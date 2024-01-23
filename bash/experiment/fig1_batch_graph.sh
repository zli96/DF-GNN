if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

convs=(gt agnn gat)
datasets=(COCO-SP PascalVOC-SP MNIST CIFAR10 CLUSTER PATTERN)
formats=(hyper)
dims=(128)
bs=1024

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for format in ${formats[@]}; do
				name=${conv}_${dataset}_${format}_dim${dim}_${Time}
				python -u DFGNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result 2>&1 | tee -a log/day_${day}/${name}.log
			done
		done
	done
done
