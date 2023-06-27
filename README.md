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

run graph transformer in ell format example 

```python
cd dgNN
python dgNN/script/test/test_gf_ell.py
```

## support dim & heads

|        | dim=64 |  bs=32  |
| :-----: | :-----: | :-----: |
| heads=1 | support | support |
| heads=2 | support | support |
| heads=8 | support |    /    |
