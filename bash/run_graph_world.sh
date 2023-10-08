# read -p "Enter num of sample(default=50): " g_range
read -p "Whether to log(default=False): " log_flag
read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter data dir(default=/workspace2/dataset): " data_dir


if [ -z "${dim}" ]; then
    dim=64
fi
if [ -z "${data_dir}" ]; then
    data_dir="/workspace2/dataset"
fi
if [ -z "${format}" ]; then
    format="csr"
fi

day=$(date +%m_%d)
# day=09_25
mkdir log/day_${day}

set -e
python setup.py develop

TASK="nodeclassification"
GENERATOR="sbm"

echo TASK $TASK
echo GENERATOR $GENERATOR


avg_degrees=(2 4 8 16 24 32 40 48 56 64)
power_exponent=9
for avg_degree in ${avg_degrees[@]}; do
    OUTPUT_PATH="/workspace2/dataset/graphworld/${TASK}_${GENERATOR}/power_ex${power_exponent}/avg_d${avg_degree}"
    if [ -n "${log_flag}" ]; then
        Time=$(date +%H_%M_%S)
        name=gf_graphworld_${format}_avgd${avg_degree}_power${power_exponent}_dim${dim}_${Time}
        python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} | tee log/day_${day}/${name}.log
    else
        python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH}
    fi
done
