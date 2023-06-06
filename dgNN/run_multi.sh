read -p "Enter dim(default=64): " dim  
read -p "Enter heads(default=8): " heads 
# read -p "Enter batch size(default=256): " bs
read -p "Enter comment: " comment

if [ -z "${dim}" ];then
	dim=64
fi
if [ -z "${heads}" ];then
	heads=8
fi
if [ -z "${comment}" ];then
	comment=normal
fi

batch_sizes=(256 128 64 32)

day=$(date +%d)
Time=$(date +%H_%M_%S)
python setup.py develop
mkdir log/day_${day}
for bs in ${batch_sizes[@]};
do
python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs  | tee log/day_${day}/weight_ver_${dim}_${heads}_${bs}_${comment}_${Time}.log
done