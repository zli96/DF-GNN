# read -p "Enter num of sample(default=50): " g_range
read -p "Whether to log(default=False): " log_flag
read -p "Enter dim(default=64): " dim
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${dim}" ]; then
    dim=64
fi
if [ -z "${data_dir}" ]; then
    data_dir="/workspace2/dataset"
fi
log_flag=1
day=$(date +%m_%d)
# day=09_25
mkdir log/day_${day}

set -e
python setup.py develop

TASK="nodeclassification"
GENERATOR="sbm"

echo TASK $TASK
echo GENERATOR $GENERATOR

formats=(csr hyper outdegree)
avg_degrees=(2 4 8 16 24 32 48 64 80 96 128 160)

power_exponent=9

Time=$(date +%H_%M_%S)

for avg_degree in ${avg_degrees[@]}; do
    for format in ${formats[@]}; do
        if [ -n "${log_flag}" ]; then
            name=gf_graphworld_${format}_avgd${avg_degree}_power${power_exponent}_dim${dim}_${Time}
            log_file=log/day_${day}/${name}.log
        else
            log_file=/dev/null
        fi
        echo format $format | tee -a $log_file
        echo avg_degree $avg_degree | tee -a $log_file
        OUTPUT_PATH="/workspace2/dataset/graphworld/${TASK}_${GENERATOR}/power_ex${power_exponent}/avg_d${avg_degree}"

        python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} --rerun 2>&1 | tee -a $log_file

    done
done
