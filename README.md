## dgNN

graph statistics [figure/graph_statistics.md](figure/graph_statistics.md)

### How to build

**creat conda env**

```
conda create -n fuse_attention
conda activate fuse_attention
conda install python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install urllib3 idna certifi matplotlib
```


```shell
git clone git@github.com:paoxiaode/fuse_attention.git
cd fuse_attention
bash install.sh
```

### Examples

We provide serval bash examples to run the model

```shell
// format: nofuse(dglsparse benchmark) 
// csr(one kernel in csr format)
// hyper (sddmm in coo + spmm in csr format)

// test bash, run the code on the molhiv dataset without the log
bash bash/run_nolog.sh 

// run the code on the different bs and print the log
bash bash/run_multi.sh 

// run the code on the full-graph dataset like cora and print the log
bash bash/run_full_graph.sh 

// profile the code by the nsight system tool
bash bash/run_nsys.sh 

// profile the code by the nsight compute tool
bash bash/run_ncu.sh 

// run the csr/hyper methods on all datasets
nohup bash bash/run_all.sh > /dev/null 2>&1 &

```

### Datasets

Current support dataset

Batch dataset: 
* mol: ogbg-molhiv, PCQM4Mv2-full
* SBM: PATTERN, CLUSTER
* superpixel： CIFAR10, MNIST
* LRGB: PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct

For Batch datasets, you can run it by [dgNN/script/test/test_gf.py](dgNN/script/test/test_gf.py)

Full dataset: (only one graph)
* cora, arxiv, pumbed, cite

For full datasets, you can run it by [dgNN/script/test/test_gf_full_graph.py](dgNN/script/test/test_gf_full_graph.py)
