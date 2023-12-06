# Protein structure prediction

## Quick Start

### Teacher-training

```python-repl
$ CUDA_VISIBLE_DEVICES=0,1 python main.py --device cuda --options pretrain --epochs 10  --save_path model/pretrain01/
```

### Student-training

```python-repl
$ CUDA_VISIBLE_DEVICES=0,1  python selftrain.py --device cuda --batch_size 64  --iteration 3 --dropout 0.3 --epochs 5 --save_path model/selftrain01/ --data_path dataset/predicted01 --loss_name GCE --gce_q 0.8 --is_homologous True
```

## Dataset

Here are some sample datasets

```python-repl
dataset/oneh/
```

```python-repl
sequence = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
```
