read -p "Whether to log(default=False): " log_flag
read -p "Whether use test mode(default=False): " test_flag
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${data_dir}" ]; then
    data_dir="/workspace2/dataset"
fi

log_flag=1
day=$(date +%m_%d)
mkdir log/day_${day}

set -e
python setup.py develop

TASK="nodeclassification"
GENERATOR="sbm"

echo TASK $TASK
echo GENERATOR $GENERATOR

if [ -n "${test_flag}" ]; then
    formats=(csr)
    avg_degrees=(2)
    dims=(32)
    echo test mode !!!!!!!!!!!!
else
    formats=(csr hyper hyper_nofuse outdegree)
    avg_degrees=(2 4 8 16 32 48 64 96 128 160)
    dims=(32 64 128)
fi
echo ${avg_degrees[@]}
power_exponent=9

Time=$(date +%H_%M_%S)
for dim in ${dims[@]}; do
    for avg_degree in ${avg_degrees[@]}; do
        for format in ${formats[@]}; do
            if [ -n "${log_flag}" ]; then
                name=gf_graphworld_power${power_exponent}_dim${dim}_avgd${avg_degree}_${format}_${Time}
                log_file=log/day_${day}/${name}.log
            else
                log_file=/dev/null
            fi
            echo format $format | tee -a $log_file
            echo avg_degree $avg_degree | tee -a $log_file
            OUTPUT_PATH="/workspace2/dataset/graphworld/${TASK}_${GENERATOR}/power_ex${power_exponent}/avg_d${avg_degree}"
            if [ -n "${test_flag}" ]; then
                ## test without log
                python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} --rerun | tee -a test.log
            else
                python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} --rerun --store-result 2>&1 | tee -a $log_file
            fi
        done
    done
done
