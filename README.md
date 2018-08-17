# ICNet

A real-time segmentation framwork in ICNet. this repo trained on Cityscape with a good perfermance. Here is some result:

![](https://i.loli.net/2018/08/16/5b74f45c3237c.png)
![](https://i.loli.net/2018/08/16/5b74f507e2304.png)
![](https://i.loli.net/2018/08/16/5b74f6cca03ff.png)

# Training

First download the pretrain model from [Here](https://drive.google.com/drive/folders/132QwCK7yretEO--cK21i3YKLp0NRtle2)
Then using:

```
python3 train.py --update-mean-var --train-beta-gamma
```

**NOTE**: `--update-mean-var` and `--train-beta-gamma` must be done, if not result get very bad.

# Dataset Preperation

Using Ciryscape like training data, provide the cityscape dataset dir and train_list.list in this format:

```
leftImg8bit/train/bremen/bremen_000315_000019_leftImg8bit.png gtFine/train/bremen/bremen_000315_000019_gtFine_labelIds.png
```

# Copyright

this codes original from [here](https://github.com/hellochick/ICNet-tensorflow), thanks for `HelloChik` work.