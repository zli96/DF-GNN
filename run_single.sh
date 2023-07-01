read -p "Enter dim(default=64): " dim  
read -p "Enter heads(default=8): " heads 
read -p "Enter batch size(default=256): " bs
read -p "Enter comment(default=normal): " comment
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${dim}" ];then
	dim=64
fi
if [ -z "${heads}" ];then
	heads=8
fi
if [ -z "${bs}" ];then
	bs=256
fi
if [ -z "${comment}" ];then
	comment=normal
fi
if [ -z "${data_dir}" ];then
	data_dir="/workspace2/dataset"
fi

day=$(date +%m_%d)
Time=$(date +%H_%M_%S)

# python setup.py develop
mkdir log/day_${day}
# python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} | tee log/day_${day}/gf_${dim}_${heads}_${bs}_${comment}_${Time}.log
python -u dgNN/script/test/test_gf_ell.py --dim $dim --heads $heads --batch-size $bs --data-dir ${data_dir} | tee log/day_${day}/gf_ell_${dim}_${heads}_${bs}_${comment}_${Time}.log
