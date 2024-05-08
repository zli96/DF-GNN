# DFGNN
DFGNN provides several fusion implementations for graph convolution

**Baseline:**
* [pyg](https://pytorch-geometric.readthedocs.io/en/latest/index.html): gnn message passing lib
* [dgl sparse](https://doc.dgl.ai/en/latest/api/python/dgl.sparse_v0.html): gnn sparse operator lib 
* [dgNN](https://github.com/dgSPARSE/dgNN): fused gnn node-parallel kernel introduced in 


**Our method:**
* tiling: one kernel with column tiling method, support for graphs with super node
* softmax: two kernels, edge-parallel sddmm kernel and node-parallel softmax+spmm kernel
* hyper: one kernel in csr+coo hyper format, support for batch graphs

## How to build

**creat conda env**

``` bash
conda create -n DFGNN
conda activate DFGNN
conda install **python**=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install urllib3 idna certifi matplotlib
```

**create docker env**

``` bash
## run pyg docker 
cd docker
bash run.sh

## build dgl
cd /workspace2/dgl
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j4  
```

**install DFGNN**

``` bash
cd DFGNN
bash install.sh
```

## Examples

We provide serval bash examples to run the model

**Measure the DFGNN kernel performance**
``` bash
# run the gt convolution on PATTERN dataset with hyper method
python -u DFGNN/script/test/test_batch_graph.py --dim 64 --batch-size 1024 --dataset PATTERN --format hyper --conv gt


# run the DFGNN on the batch graph datasets
bash bash/run_batch_graph.sh

# run the DFGNN on the full graph datasets
bash bash/run_full_graph.sh 

# run the DFGNN on the full graph with super node datasets
bash bash/run_full_graph_super_node.sh

# profile the code by the nsight system tool
bash bash/run_nsys.sh 

# profile the code by the nsight compute tool
bash bash/run_ncu.sh 
```

**Measure the DFGNN training performance**
``` bash
# Batch graph datasets
bash bash/run_batch_graph_train_timing.sh

# Full graph datasets
bash bash/run_full_graph_train_timing.sh
```

### Datasets

Current support dataset

Batch dataset: 
* mol: ogbg-molhiv, PCQM4Mv2-full
* SBM: PATTERN, CLUSTER
* superpixel： CIFAR10, MNIST
* LRGB: PascalVOC-SP, COCO-SP, Peptides-func, Peptides-struct


Full dataset: (only one graph)
* cora, arxiv, pumbed, cite

Full dataset with super node: (only one graph)
* ppa, reddit, protein