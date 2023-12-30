read -p "Whether use test mode(default=False): " test_flag
export alblation_mode="3"

if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	convs=(gt)
	# datasets=(protein yelp Flickr AmazonCoBuyComputer AmazonCoBuyPhoto CoauthorCS CoauthorPhysics ppa collab)
	datasets=(protein yelp Flickr CoauthorCS CoauthorPhysics ppa collab)
	formats=(csr_gm hyper_ablation csr)
	formats=(tiling)

	dims=(256)
	echo test mode !!!!!!!!!!!!
else
	convs=(gt)
	datasets=(reddit)
	formats=(csr_gm hyper_ablation tiling)
	dims=(512)
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
