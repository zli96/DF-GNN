## dgNN


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
bash run_nolog.sh
```

<!-- Our training script is modified from [DGL](https://github.com/dmlc/dgl). Now we implements three popular GNN models.

**Run GAT**

[DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)

```python
cd dgNN/script/train
python train_gatconv.py --num-hidden=64 --num-heads=4 --dataset cora --gpu 0
```

**Run Monet**

[DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/monet)

```python
cd dgNN/script/train
python train_gmmconv.py --n-kernels 3 --pseudo-dim 2 --dataset cora --gpu 0
```

**Run PointCloud**

We use modelnet40-sampled-2048 data in our PointNet. [DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pointcloud)

```python
cd dgNN/script/train
python train_edgeconv.py
```

### Collaborative Projects

[CogDL](https://github.com/THUDM/cogdl) is a flexible and efficient graph-learning framework that uses GE-SpMM to accelerate GNN algorithms. This repo is implemented in CogDL as a submodule. -->
