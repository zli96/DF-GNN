# # available parameter
# datasets=(PATTERN CLUSTER MNIST CIFAR10 Peptides-func COCO-SP PascalVOC-SP)
# formats=(csr softmax hyper)
# batch_sizes=(16 32 64 128 256 512 1024 2048 4096)

convs=(gt)
datasets=(PATTERN)
formats=(hyper)
batch_sizes=(1024)
dims=(32 64 128)

data_dir="/workspace2/dataset"

set -e
python setup.py develop
for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for format in ${formats[@]}; do
				for bs in ${batch_sizes[@]}; do
					python -u DFGNN/script/test/test_batch_graph.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv}
				done
			done
		done
	done
done
