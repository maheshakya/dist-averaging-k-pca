# dist-averaging-k-pca

## Requirements:
	- Python 2.7
	- numpy
	- scipy
	- scikit-learn
	- matplotlib
	- pandas
	- sacred (https://github.com/IDSIA/sacred) - for reproducibility

## How to run experiments:

Set the working directory to: sacred/

For synthetic data experiments run: python synthetic_data_exp.py

For real data experiments:

1. mnist small dataset
Run: python mnist_small_exps.py

2. NIPS papers dataset
Download the file at: https://archive.ics.uci.edu/ml/machine-learning-databases/00371/NIPS_1987-2015.csv and place it in sacrad/ directory.
Run: python nips_data_exps.py

3. FMA music dataset
Download the  fma_metadata.zip file from: https://github.com/mdeff/fma
Extract features.csv file from fma_metadata.zip into sacred/ directory.
Run: python fma_exps.py
