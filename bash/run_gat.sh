read -p "Whether use test mode(default=False): " test_flag

conv=gat

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	datasets=(PATTERN CLUSTER ogbg-molhiv MNIST CIFAR10)
	formats=(hyper csr)
	batch_sizes=(16)
	dims=(128)
	echo test mode !!!!!!!!!!!!
else
	datasets=(ogbg-molhiv PATTERN CLUSTER MNIST CIFAR10 Peptides-func Peptides-struct PascalVOC-SP COCO-SP)
	formats=(csr hyper softmax indegree)
	batch_sizes=(16 32 64 128 256 512 1024 2048 4096)
	dims=(32 64 128)
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
				# # run with log

				# echo "nohup python -u dgNN/script/test/test_gt.py --dim $dim --heads $heads --batch-size $bs  > log/day_${day}/gt_${dim}_${heads}_${bs}_${comment}_${Time}.log 2>&1 &" | bash;
			done
		done
	done
done
