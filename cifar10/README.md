# CIFAR10 Experiments
This folder contains files we used for CIFAR10 experiments.

* `make_data.py` downloads and makes training and testing datasets. 
* `create_label_errors.ipynb` creates label errors of various noise and sparsity levels.
* `generate_commands.ipynb` generates commands for NN training.
* `cifar10_train.py` trains NNs. Please refer to the comments in `__main__` for examples of commands.
* `cifar10_mask.py` computes masks for both CTRL and CL. It also calculates mask accuracy. To use the code, set variables in the `###` block and then run with `python cifar10_mask.py`. It outputs `mask_param_search.csv` if the clustering parameters search flag is set to True, outputs masks and `mask_results.csv` if the mask compute flag is turned on.
* `cifar10_evaluate.py` computes test accuracies for models retrained by various methods. To use the code, set variables in the `###` block and then run with `python cifar10_evaluate.py`. It outputs an `evaluation.csv`.
* `cifar10_results.ipynb` contains examples of decoding the `.csv` files.
* `models` contains codes for building ResNets.
* `files.txt` shows folders and files arrangements.

## Steps to run
* Run `python make_data.py`, it creates `train` and `test` folders in `datasets/datasets/cifar10/cifar10`.
* Specify noise and sparsity levels and then run `create_label_errors.ipynb` which creates and writes noisy labels into `cifar10_noisy_labels`.
* Train NNs using noisy labels. The `# training` cell in the `generate_commands.ipynb` file can generate a list of commands for creating folders and running `cifar10_train.py`. Use make directories commands to create folders first, and then, train NNs. There are two ways to call `cifar10_train.py` where instructions can be found in `__main__`. During NN training, `cifar10_train.py` writes checkpoints and logs into `cifar10_training`, and loss matrices into `cifar10_training_loss`.
* If want to test Confident Learning (CL) methods as well, can use the `# train and combine cv` cell to generate commands for folder creation, cross-validated NN training, and prediction combination. After training, extra files of prediction probabilities will be saved in the `cifar10_training` folder.
* Compute masks and masks statistics by running `python cifar10_mask.py`. It saves masks in `cifar10_mask` (and `cifar10_mask_cl` for CL methods).
* Retrain NNs with the masked labels. Use the `# retraining` cell in `generate_commands.ipynb` to generate retraining commands for running `cifar10_train.py`. Checkpoints and logs will be saved in `cifar10_training_masked`. In this step, CTRL and CL use similar commands.
* Evaluate retrained models by running `python cifar10_evaluate.py`.

* `files.txt` outlines structures of the outputs from the above steps.
* The `# training with pruning`, `# retraining with static model-pred labels`, and `# retraining with dynamic model-pred labels` cells in `generate_commands.ipynb` generate commands for ablation studies.


## References
* [Confident Learning CIFAR10 Reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10)
* [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)
