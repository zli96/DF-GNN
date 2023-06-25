read -p "Enter dim(default=64): " dim  
read -p "Enter heads(default=8): " heads 
read -p "Enter batch size(default=256): " bs

if [ -z "${dim}" ];then
	dim=64
fi
if [ -z "${heads}" ];then
	heads=8
fi
if [ -z "${bs}" ];then
	bs=256
fi


python setup.py develop
# python -u dgNN/script/test/test_gf.py --dim $dim --heads $heads --batch-size $bs
python -u dgNN/script/test/test_gf_ell.py --dim $dim --heads $heads --batch-size $bs
