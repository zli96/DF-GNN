read -p "Whether use test mode(default=False): " test_flag

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	conv=gt
	datasets=(PATTERN)
	formats=(hyper)
	batch_sizes=(1024)
	dims=(32 64)
	echo test mode !!!!!!!!!!!!
else
	datasets=(PATTERN CLUSTER MNIST CIFAR10 Peptides-func COCO-SP PascalVOC-SP)
	formats=(csr hyper softmax tiling)
	batch_sizes=(16 32 64 128 256 512 1024 2048 4096)
	dims=(64)

fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for format in ${formats[@]}; do
			name=${conv}_${dataset}_${format}_dim${dim}_${Time}
			for bs in ${batch_sizes[@]}; do
				if [ -n "${test_flag}" ]; then
					# # run with nolog
					python -u dgNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv}
				else
					python -u dgNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result 2>&1 | tee -a log/day_${day}/${name}.log

				fi
			done
		done
	done
done
