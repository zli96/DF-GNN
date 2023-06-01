# read -p "Enter mode:(0:normal, 1:debug)" mode  
# if [ $mode == 1 ];
# then
# echo debug mode
# python -m pdb dgNN/script/train/train_gatconv.py --n-hidden=64 --n-heads=4 --dataset cora --gpu 0 --n-layers 2
# else
# echo normal run
# python  dgNN/script/train/train_gatconv.py --n-hidden=64 --n-heads=4 --dataset cora --gpu 0 --n-layers 2
# fi
read -p "Enter dim: " dim  
read -p "Enter heads: " heads 


day=$(date +%d)
Time=$(date +%H_%M_%S)

python setup.py develop
mkdir log/day_${day}
python dgNN/script/test/test_gf.py --dim $dim --heads $heads  | tee log/day_${day}/weight_ver_${dim}_${heads}_${Time}.log
