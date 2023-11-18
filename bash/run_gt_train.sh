conv=gt

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

datasets=(ogbg-molhiv)
formats=(hyper)
batch_sizes=(256)
dims=(64)
echo test mode !!!!!!!!!!!!

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for dim in ${dims[@]}; do
	for dataset in ${datasets[@]}; do
		for format in ${formats[@]}; do
			for bs in ${batch_sizes[@]}; do
				python -u dgNN/script/train/train_gtconv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset}

				# python -u dgNN/script/train/train_gtconv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --checkgrad

			done
		done
	done
done
