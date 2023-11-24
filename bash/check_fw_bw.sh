read -p "Whether to log(default=False): " log_flag

if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop
if [ -n "${test_flag}" ]; then
	log=log/check_fw_bw.log
else
	log=/tmp/null
fi
## forward check

## GT
python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format hyper --conv gt | tee $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format csr --conv gt | tee -a $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format softmax --conv gt | tee -a $log

## DOTGAT
python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format hyper --conv dotgat | tee -a $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format csr --conv dotgat | tee -a $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format softmax --conv dotgat | tee -a $log

## GAT
python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format hyper --conv gat | tee -a $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format csr --conv gat | tee -a $log

python -u dgNN/script/test/test_fuse_conv.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --format softmax --conv gat | tee -a $log

## backward check
python -u dgNN/script/train/test_gtconv_fw_bw.py --dim 64 --batch-size 64 --data-dir ${data_dir} --dataset PATTERN --checkgrad | tee -a $log
