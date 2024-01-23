read -p "Whether use test mode(default=False): " test_flag

if [ -z "${data_dir}" ]; then
	data_dir="/workspace2/dataset"
fi

if [ -n "${test_flag}" ]; then
	# convs=(gt)
	# # datasets=(protein yelp Flickr AmazonCoBuyComputer AmazonCoBuyPhoto CoauthorCS CoauthorPhysics ppa collab)
	# datasets=(cora pubmed cite)
	# formats=(csr)
	# dims=(512)
	# echo test mode !!!!!!!!!!!!

	# feature parallel && edge parallel && node parallel
	# export alblation_mode="4"
	# convs=(gt)
	# datasets=(arxiv)
	# formats=(csr hyper_ablation hyper)
	# dims=(64)
	convs=(gat)
	datasets=(reddit protein)
	# datasets=(arxiv)
	# formats=(tiling)
	formats=(softmax_gm tiling)

	# formats=(softmax_gm)
	dims=(128)
	echo test mode !!!!!!!!!!!!
else
	# convs=(gt agnn gat)
	# datasets=(cora pubmed cite)
	# formats=(csr hyper softmax)
	# dims=(64)

	# convs=(gt)
	# datasets=(protein yelp Flickr AmazonCoBuyComputer AmazonCoBuyPhoto CoauthorCS CoauthorPhysics ppa collab)
	# formats=(csr)
	# dims=(128 256 512)

	convs=(gt)
	datasets=(Flickr reddit protein)
	formats=(csr_gm tiling softmax_gm)
	dims=(128)
	echo test mode !!!!!!!!!!!!
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
					python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv}
				else
					python -u DFGNN/script/test/test_full_graph.py --dim $dim --dataset ${dataset} --data-dir ${data_dir} --format ${format} --conv ${conv} --store-result 2>&1 |
						tee log/day_${day}/${name}.log
				fi
			done
		done
	done
done
