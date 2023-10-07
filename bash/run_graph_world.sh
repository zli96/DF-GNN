read -p "What config to shmoo(default=power_exponent): " shmoo
read -p "Whether to log(default=False): " log_flag
read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter graph range(default=100): " g_range
read -p "Enter data dir(default=/workspace2/dataset): " data_dir

if [ -z "${shmoo}" ]; then
    shmoo="power_exponent"
fi

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




if [ "$shmoo" == "power_exponent" ]; then
    max_degrees=(9)
    for max_d in ${max_degrees[@]}; do
        OUTPUT_PATH="/workspace2/dataset/graphworld/${TASK}_${GENERATOR}/${shmoo}/${max_d}"
        if [ -n "${log_flag}" ]; then
            Time=$(date +%H_%M_%S)
            name=gf_graphworld_${shmoo}_${max_d}_${format}_dim${dim}_${Time}
            python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} | tee log/day_${day}/${name}.log
        else
            python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH}
        fi
    done
fi
