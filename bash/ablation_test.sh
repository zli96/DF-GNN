conv=gt

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

dataset=PATTERN
format=hyper_ablation
bs=1024
dims=(32 64 128 256 512)
modes=("0" "1" "2")

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for dim in ${dims[@]}; do
	name=${conv}_${dataset}_${format}_dim${dim}_${Time}
	for mode in ${modes[@]}; do
		export alblation_mode=$mode
		python -u dgNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} | tee -a log/day_${day}/${name}.log
	done
	python -u dgNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format hyper --conv ${conv} | tee -a log/day_${day}/${name}.log
done
