# # available parameter
# convs=(gt agnn gat)
# datasets=(ppa reddit protein)
# formats=(csr_gm softmax_gm tiling)

convs=(gt)
datasets=(ppa)
formats=(csr_gm softmax_gm tiling)
dims=(128)

data_dir="/workspace2/dataset"

set -e
python setup.py develop

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for format in ${formats[@]}; do
				python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv}
			done
		done
	done
done
