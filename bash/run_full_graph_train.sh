read -p "Whether use test mode(default=False): " test_flag

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	datasets=(cora)
	dim=64
	echo test mode !!!!!!!!!!!!
else
	datasets=(cora cite pubmed)
	dim=64
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

## train
for dataset in ${datasets[@]}; do
	if [ -n "${test_flag}" ]; then
		python -u dgNN/script/train/train_gtconv_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --n-epochs 100
	else
		python -u dgNN/script/train/train_gtconv_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --n-epochs 100 | tee log/day_${day}/gt_train_full_graph_${dataset}_dim${dim}.log
	fi
done
