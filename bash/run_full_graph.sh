read -p "Whether use test mode(default=False): " test_flag

conv=

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	convs=(gt)
	datasets=(arxiv)
	formats=(hyper)
	dims=(64)
	echo test mode !!!!!!!!!!!!
else
	datasets=(cora pubmed cite reddit)
	formats=(csr hyper softmax)
	batch_sizes=(16 32 64 128 256 512 1024 2048 4096)
	dims=(32 64 128)
fi

# formats=(csr hyper)
day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for format in ${formats[@]}; do
			name=${conv}_${dataset}_${format}_dim${dim}_${Time}
			if [ -n "${test_flag}" ]; then
				# # run with nolog
				python -u dgNN/script/test/test_gt_full_graph.py --dim $dim --heads $heads --dataset ${dataset} --data-dir ${data_dir} --format ${format}
			else
				python -u dgNN/script/test/test_gt_full_graph.py --dim $dim --heads $heads --dataset ${dataset} --data-dir ${data_dir} --format ${format} | tee log/day_${day}/gt_${dataset}_${format}_dim${dim}_h${heads}_${Time}.log
			fi
		done
	done
done
