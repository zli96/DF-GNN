read -p "Enter dataset(default=ogbg-molhiv): " dataset
read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter heads(default=1): " heads
read -p "Enter batch size(default=256): " bs
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${dim}" ]; then
	dim=64
fi
if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${bs}" ]; then
	bs=256
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi
if [ -z "${dataset}" ]; then
	dataset="ogbg-molhiv"
	# dataset="PATTERN"
	# dataset="Peptides-func"
	# dataset="Peptides-struct"
	# dataset="PascalVOC-SP"
	# dataset="COCO-SP"
fi
if [ -z "${format}" ]; then
	format="csr"
	# format="hyper"
	# format="outdegree"
fi

set -e

python setup.py develop
python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format}
# python -u dgNN/script/test/test_gf_full_graph.py --dim $dim --heads $heads --dataset pubmed --data-dir ${data_dir} --format ${format}
# cuda-gdb -ex r --args  python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format}
# CUDA_LAUNCH_BLOCKING=1 python -u dgNN/script/test/test_gf.py --data-dir ${data_dir} --format ${format} --config ${config_dir} --batch-size $bs