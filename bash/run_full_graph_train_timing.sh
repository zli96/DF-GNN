# # available parameter
# convs=(gt agnn)
# datasets=(cora cite pubmed)

convs=(gt)
datasets=(cora cite pubmed)
dims=(64)

data_dir="/workspace2/dataset"

set -e
python setup.py develop

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			python -u DFGNN/script/train/train_full_graph_timing.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --n-epochs 100
		done
	done
done
