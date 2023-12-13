read -p "Whether use test mode(default=False): " test_flag

if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	convs=(agnn)
	datasets=(cora pubmed cite)
	formats=(hyper)
	dims=(32)
	echo test mode !!!!!!!!!!!!
else
	convs=(gt agnn gat)
	datasets=(cora pubmed cite)
	formats=(csr hyper softmax)
	dims=(64)
fi

# formats=(csr hyper)
day=$(date +%m_%d)
Time=$(date +%H_%M_%S)
mkdir log/day_${day}

set -e
python setup.py develop

for conv in ${convs[@]}; do
	for dim in ${dims[@]}; do
		for dataset in ${datasets[@]}; do
			for format in ${formats[@]}; do
				name=${conv}_${dataset}_${format}_dim${dim}_${Time}
				if [ -n "${test_flag}" ]; then
					# # run with nolog
					python -u dgNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv}
				else
					python -u dgNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv} --store-result 2>&1 |
						tee log/day_${day}/${name}.log
				fi
			done
		done
	done
done
