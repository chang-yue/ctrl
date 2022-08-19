# CTRL: Clustering Training Losses for Label Error Detection
This repo contains the implementation of CTRL and files for running experiments reported in the paper.

* `clustering.py` computes masks using CTRL.
* `confident_learning.py` computes masks using [Confident Learning](https://arxiv.org/abs/1911.00068) methods.
* `cifar10` contains files for CIFAR10 experiments.
* `cifar100` contains instructions of how to modify files in cifar10 to conduct experiments for CIFAR100.
* `tabular` contains files for tabular datasets experiments.

Please cite [our paper](https://arxiv.org/abs/2208.08464) if you use this package.
