read -p "Whether generate new data(default=False): " gen_flag
read -p "Whether to log(default=False): " log_flag
read -p "Enter format(default=csr): " format
read -p "Enter dim(default=64): " dim
read -p "Enter graph range(default=100): " g_range
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

TASK="nodeclassification"
GENERATOR="sbm"
function get_task_class_name() {
    local task=$1
    case $task in
    nodeclassification) echo "NodeClassification" ;;
    graphregression) echo "GraphRegression" ;;
    linkprediction) echo "LinkPrediction" ;;
    noderegression) echo "NodeRegression" ;;
    *) echo "BAD_BASH_TASK_INPUT_${task}_" ;;
    esac
}

while getopts t:g: flag; do
    case "${flag}" in
    t) TASK=${OPTARG} ;;
    g) GENERATOR=${OPTARG} ;;
    esac
done

echo TASK $TASK
echo GENERATOR $GENERATOR

day=$(date +%m_%d)
OUTPUT_PATH="/workspace2/dataset/graphworld/day_${day}/${TASK}_${GENERATOR}"
if [ -n "${gen_flag}" ]; then
    NUM_SAMPLES=5

    rm -rf "${OUTPUT_PATH}"
    mkdir -p ${OUTPUT_PATH}

    # # Add gin file string.
    GIN_FILES="/workspace2/graphworld/src/configs/${TASK}.gin "
    GIN_FILES="${GIN_FILES} /workspace2/graphworld/src/${TASK}_generators/${GENERATOR}/default_setup.gin"
    GIN_FILES="${GIN_FILES} /workspace2/graphworld/src/configs/common_hparams/${TASK}_test.gin"

    # # Add gin param string.
    # TASK_CLASS_NAME=$(get_task_class_name ${TASK})
    GIN_PARAMS="GeneratorBeamHandlerWrapper.nsamples=${NUM_SAMPLES}"

    python3 /workspace2/graphworld/src/beam_benchmark_main.py \
        --runner DirectRunner \
        --gin_files configs/DGL_${TASK}.gin \
        --gin_params ${GIN_PARAMS} \
        --output ${OUTPUT_PATH} --write_intermediat True
fi


if [ -n "${log_flag}" ]; then
    mkdir log/day_${day}
    Time=$(date +%H_%M_%S)
    name=gf_graphworld_${format}_dim${dim}_${Time}
    python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH} | tee log/day_${day}/${name}.log
else
    python -u dgNN/script/test/test_gf_graphworld.py --dim $dim --data-dir ${data_dir} --format ${format} --output ${OUTPUT_PATH}
fi
