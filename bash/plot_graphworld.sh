data_dir="/workspace2/dataset"

TASK="nodeclassification"
GENERATOR="sbm"

echo TASK $TASK
echo GENERATOR $GENERATOR

power_exponent=9
dims=(32 64 128)

OUTPUT_PATH="/workspace2/dataset/graphworld/${TASK}_${GENERATOR}/power_ex${power_exponent}"
for dim in ${dims[@]}; do
	python -u dgNN/utils/graphworld_statistics.py --dim $dim --data-dir ${data_dir} --output ${OUTPUT_PATH}
done
