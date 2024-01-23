# DFGNN
DFGNN provides several fusion implementations for graph convolution

**Baseline:**
* nofuse: dglsparse package in DGL lib without fusion
* csr: one kernel in csr format
* softmax: two kernels, sddmm in coo and softmax+spmm in c

**Our method:**
* tiling: one kernel with column tiling method, support for graphs with super node
* hyper: one kernel in csr+coo hyper format, support for batch graphs

## How to build

**creat conda env**

```
conda create -n DFGNN
conda activate DFGNN
conda install **python**=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install urllib3 idna certifi matplotlib
```


```shell
git clone git@github.com:paoxiaode/DFGNN.git
cd DFGNN
bash install.sh
```

## Examples

We provide serval bash examples to run the model

```shell
// run the gt convolution on PATTERN dataset with hyper method
python -u DFGNN/script/test/test_fuse_conv.py --dim 64 --batch-size 1024 --dataset PATTERN --format hyper --conv gt


// run the DFGNN on the batch graph datasets
bash bash/run_batch_graph.sh

// run the DFGNN on the full graph datasets
bash bash/run_full_graph.sh 

// run the DFGNN on the full graph with super node datasets
bash bash/run_full_graph_super_node.sh

// profile the code by the nsight system tool
bash bash/run_nsys.sh 

// profile the code by the nsight compute tool
bash bash/run_ncu.sh 
```

### Datasets

Current support dataset

Batch dataset: 
* mol: ogbg-molhiv, PCQM4Mv2-full
* SBM: PATTERN, CLUSTER
* superpixel： CIFAR10, MNIST
* LRGB: PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct

For Batch datasets, you can run it by [dgNN/script/test/test_gt.py](dgNN/script/test/test_gt.py)

Full dataset: (only one graph)
* cora, arxiv, pumbed, cite

For full datasets, you can run it by [dgNN/script/test/test_gt_full_graph.py](dgNN/script/test/test_gt_full_graph.py)