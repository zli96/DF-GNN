# Fuse attention

## dgNN

build dgNN

```bash
cd dgNN
bash install.sh
##############
python -W ignore setup.py develop
```

run gat example

```python
cd dgNN/script/train
python train_gatconv.py --n-hidden=64 --n-heads=4 --dataset cora --gpu 0
```
