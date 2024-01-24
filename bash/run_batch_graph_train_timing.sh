# # available parameter
# convs=(gt agnn)
# datasets=(PATTERN CLUSTER COCO-SP PascalVOC-SP)
# formats=(hyper)

convs=(gt)
datasets=(PATTERN)
formats=(hyper)
batch_sizes=(64)
dims=(64)

data_dir="/workspace2/dataset"

set -e
python setup.py develop
for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for bs in ${batch_sizes[@]}; do
				python -u DFGNN/script/train/train_batch_graph_timing.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --conv ${conv}
			done
		done
	done
done
