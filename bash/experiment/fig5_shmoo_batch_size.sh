if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
log_dir=log_new/day_${day}/abl_batch
mkdir -p $log_dir

set -e
python setup.py develop

conv=gt
datasets=(PATTERN)
formats=(csr cugraph hyper)
dim=128
batch_sizes=(64 128 256 512 1024 2048)
name=fig5_shmoo_batch_size_${Time}

for dataset in ${datasets[@]}; do
	for format in ${formats[@]}; do
		for bs in ${batch_sizes[@]}; do
			python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result 2>&1 | tee -a $log_dir/${name}.log
		done
	done
done
