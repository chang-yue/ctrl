# Tabular Experiments
This folder contains files we used for tabular datasets experiments.

* `data_analysis` contains one `.ipynb` data analysis notebook per each dataset.
* `prepare_datasets.ipynb` preprocesses datasets and create noisy labels.
* `test.py` can search NN training hyperparameters (outputs `model_params_search.csv` and `model_params.csv`), train/retrain models, compute and evaluate masks (outputs mask files and `mask_results.csv`), evaluate model test accuracy (outputs `evaluation.csv`). To run the code, set flags and the corresponding variables in the `###------###` block and then run with `python test.py`.
* `neural_net.py` contains functions for NN training and model evaluation.
* `utils.py` contains helper functions for data processing and NN training.
* `tabular_results_20.ipynb` contains examples of decoding the `.csv` files.
* `files.txt` shows folders and files arrangements.
