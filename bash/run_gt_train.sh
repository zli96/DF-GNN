conv=gt

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

# datasets=(PATTERN)
datasets=(PascalVOC-SP COCO-SP CLUSTER)

formats=(hyper)
batch_sizes=(1024)
dims=(64)

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

# for dim in ${dims[@]}; do
# 	for bs in ${batch_sizes[@]}; do
# 		name=train_gtconv_dim${dim}_bs${bs}
# 		python -u dgNN/script/train/train_gtconv.py --dim $dim --batch-size $bs --data-dir ${data_dir} | tee -a log/day_${day}/${name}.log
# 	done
# done

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for bs in ${batch_sizes[@]}; do
			name=train_gtconv_${dataset}_dim${dim}_bs${bs}
			python -u dgNN/script/train/test_gtconv_fw_bw.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} | tee log/day_${day}/${name}.log
			# python -u dgNN/script/train/test_gtconv_fw_bw.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset}
		done
	done
done
