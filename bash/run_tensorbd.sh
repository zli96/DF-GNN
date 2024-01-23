if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

conv=dotgat
dataset=PATTERN
format=nofuse
bs=64
dim=64

set -e
python setup.py develop

python -u DFGNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --profile
