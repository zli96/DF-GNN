export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="4"
export PROF_TARGET_RANGE="2"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"

read -p "Enter dim(default=64): " dim
read -p "Enter heads(default=1): " heads
read -p "Enter data dir(default=/workspace2/dataset): " data_dir
read -p "Enter dataset(ogbg-molhiv): " dataset

if [ -z "${dim}" ]; then
	dim=64
fi
if [ -z "${heads}" ]; then
	heads=1
fi
if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi
if [ -z "${dataset}" ]; then
	dataset="ogbg-molhiv"
fi

day=$(date +%m_%d)
mkdir log/nsys/day_${day}

set -e
python setup.py develop

format=nofuse
conv=dotgat
# batch_sizes=(32 64 128 256 512 1024 2048 4096)
batch_sizes=(32)
logtime=$(date +%m_%d_%H_%M_%S)
for bs in ${batch_sizes[@]}; do
	name=nsys_${conv}_${format}_dim${dim}_head${heads}_bs${bs}_${logtime}
	nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o log/nsys/day_${day}/${name} --stats true --force-overwrite true -c cudaProfilerApi --kill none python -u dgNN/script/test/test_fuse_conv.py --dim $dim --batch-size $bs --data-dir ${data_dir} --dataset ${dataset} --format ${format} --conv ${conv} --profile >log/nsys/day_${day}/${name}.log 2>&1

done
