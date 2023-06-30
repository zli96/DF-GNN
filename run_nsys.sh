
export PROF_TARGET_PASS="NVPROF"
export PROF_TARGET_SESSION="4"
export PROF_TARGET_RANGE="2"
export PROF_EARLY_EXIT=true
export PYTHONPATH="/workspace2/python_profiler/:$PYTHONPATH"

read -p "Enter dim(default=64): " dim  
read -p "Enter heads(default=8): " heads 
# read -p "Enter batch size(default=256): " bs

if [ -z "${dim}" ];then
	dim=64
fi
if [ -z "${heads}" ];then
	heads=8
fi
# if [ -z "${bs}" ];then
# 	bs=256
# fi

batch_sizes=(32 64 128 256 512 1024 2048 4096)
logtime=$(date +%H_%M_%S)
for bs in ${batch_sizes[@]};
do
name=nsys_gf_nofuse_dim${dim}_head${heads}_bs${bs}_${logtime}
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o log/nsys/${name} --stats true --force-overwrite true -c cudaProfilerApi --kill none python -u dgNN/script/test/test_gf_profile.py --dim $dim --heads $heads --batch-size $bs > log/nsys/${name}.log 2>&1 
done