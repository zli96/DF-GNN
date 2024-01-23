export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="4"
export PROF_TARGET_RANGE="2"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"

conv=gt
format=hyper
dim=128
heads=1
data_dir="/workspace2/dataset"

day=$(date +%m_%d)
mkdir log/ncu/day_${day}

set -e
python setup.py develop

# batch_sizes=(32 64 128 256 512 1024 2048 4096)
# batch_sizes=(2048)
# logtime=$(date +%m_%d_%H_%M_%S)
# for bs in ${batch_sizes[@]};
# do
# name=ncu_gat_${format}_dim${dim}_head${heads}_bs${bs}_${logtime}
# ncu --set full --import-source yes -c 10 -o log/ncu/day_${day}/${name} -k "fused_inference_kernel_hyper" python DFGNN/script/test/test_gat.py --format ${format} --dim $dim --heads $heads --batch-size $bs --dataset ${dataset} --data-dir ${data_dir} --profile > log/ncu/day_${day}/${name}.log 2>&1
# done

batch_sizes=(1024)
datasets=(PATTERN)
formats=(hyper)

logtime=$(date +%m_%d_%H_%M_%S)
for dataset in ${datasets[@]}; do
	for bs in ${batch_sizes[@]}; do
		for format in ${formats[@]}; do
			name=ncu_${conv}_${format}_${dataset}_dim${dim}_head${heads}_bs${bs}_${logtime}
			if [ "$format" == "csr" ]; then
				ncu --set full --import-source yes -c 10 -o log/ncu/day_${day}/${name} -k "fused_gt_csr" python DFGNN/script/test/test_fuse_conv.py --format ${format} --dim $dim --heads $heads --batch-size $bs --dataset ${dataset} --data-dir ${data_dir} --conv ${conv} --profile >log/ncu/day_${day}/${name}.log 2>&1
			else
				ncu --set full --import-source yes -c 10 -o log/ncu/day_${day}/${name} -k "fused_gt_hyper_inference" python DFGNN/script/test/test_fuse_conv.py --format ${format} --dim $dim --heads $heads --batch-size $bs --dataset ${dataset} --data-dir ${data_dir} --conv ${conv} --profile >log/ncu/day_${day}/${name}.log 2>&1
			fi
		done
	done
done
