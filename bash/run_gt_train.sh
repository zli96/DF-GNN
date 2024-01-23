read -p "Whether use test mode(default=False): " test_flag

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	conv=gt
	datasets=(PascalVOC-SP)
	formats=(hyper)
	batch_sizes=(1024)
	dims=(64)
	echo test mode !!!!!!!!!!!!
else
	conv=gt
	datasets=(PATTERN CLUSTER PascalVOC-SP COCO-SP)
	formats=(hyper)
	batch_sizes=(128)
	dims=(64)
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

# for dim in ${dims[@]}; do
# 	for bs in ${batch_sizes[@]}; do
# 		# name=train_gtconv_dim${dim}_bs${bs}
# 		# python -u DFGNN/script/train/train_gtconv.py --dim $dim --batch-size $bs --data-dir ${data_dir} | tee -a log/day_${day}/${name}.log
# 		python -u DFGNN/script/train/train_gtconv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --checkgrad
# 	done
# done

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for bs in ${batch_sizes[@]}; do
			if [ -n "${test_flag}" ]; then
				# run with nolog
				python -u DFGNN/script/train/test_conv_fw_bw.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --conv ${conv}
			else
				name=train_${conv}conv_${dataset}_dim${dim}_bs${bs}
				python -u DFGNN/script/train/test_conv_fw_bw.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --conv ${conv} | tee log/day_${day}/${name}.log
			fi
		done
	done
done
