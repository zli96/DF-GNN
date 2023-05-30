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
cd dgNN/dgNN/script/train
python train_gatconv.py --n-hidden=64 --n-heads=4 --dataset cora --gpu 0
```

run graph transformer example

```python
cd dgNN
python dgNN/script/test/test_gf.py
```