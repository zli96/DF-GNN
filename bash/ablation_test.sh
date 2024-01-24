conv=gt

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

datasets=(PATTERN)
# datasets=(ogbg-molhiv)

format=hyper_ablation
bs=1024
# dims=(64 128 160 192 224 256)
dims=(256)

modes=("0" "1" "2")

# dims=(256)
# modes=("3")

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

# for dataset in ${datasets[@]}; do
# 	for dim in ${dims[@]}; do
# 		# name=${conv}_${dataset}_${format}_dim${dim}_${Time}
# 		name=${conv}_${dataset}_${format}_dim${dim}_${Time}

# 		for mode in ${modes[@]}; do
# 			export alblation_mode=$mode
# 			python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} | tee -a log/day_${day}/${name}.log
# 		done
# 		python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format hyper --conv ${conv} --store-result | tee -a log/day_${day}/${name}.log
# 	done
# done

for dataset in ${datasets[@]}; do
	for dim in ${dims[@]}; do
		name=${conv}_${dataset}_${format}_dim${dim}_${Time}
		for mode in ${modes[@]}; do
			export alblation_mode=$mode
			python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv}
		done
		python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format hyper --conv ${conv}
	done
done
