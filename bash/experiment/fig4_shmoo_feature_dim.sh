if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
log_dir=log_new/day_${day}/abl_feature
mkdir -p $log_dir

set -e
python setup.py develop

conv=gt
datasets=(PATTERN)
formats=(csr cugraph hyper)
dims=(16 32 64 128 256)
bs=1024
name=fig4_shmoo_feature_dim_${Time}

for dataset in ${datasets[@]}; do
	for format in ${formats[@]}; do
		for dim in ${dims[@]}; do
			python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --store-result 2>&1 | tee -a $log_dir/${name}.log
		done
	done
done
