##
from MyKmeans import MyKmeans
from MyPreprocessing import MyPreprocessing
from MyPCA import MyPCA
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from config_loader import load, clf_names
import argparse
import sys
import seaborn as sns

import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


##
from Validation import validation_metrics
from Validation import best_k

##
if __name__ == '__main__':
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="clustering.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    ##
    dataset = config.get('clustering', 'dataset')
    path = 'datasets/' + dataset + '.arff'
    try:
        data, meta = loadarff(path)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path))
        sys.exit(1)

    ##
    kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    n_components = int(config.get('pca', 'n_components'))

    ## Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    # PCA
    pca = MyPCA(n_components)
    pca.fit(df)
    explained_variance = pca.n_eigval/sum(pca.eigval)
    print('Original Covariance Matrix: ')
    print(pca.cov_mat)
    print()

    print('Original eigen values and corresponding eigen vectors: ')
    print(pca.eigval)
    print(pca.eigvec)
    print()

    print('K max eigen values and corresponding eigen vectors: ')
    print(pca.n_eigval)
    print(pca.n_eigvec)
    print()

    print('Explained variance for %d components: ' %(n_components))
    print(np.cumsum(explained_variance))
    print()

    print('PCA sklearn algorithm ')
    pca = PCA(n_components=n_components)
    pca.fit_transform(df)
    print(np.cumsum(pca.explained_variance_ratio_))
    print(pca.singular_values_)
    print()

    print('IncrementalPCA sklearn  algorithm ')
    pca = IncrementalPCA(n_components=n_components)
    pca.fit_transform(df)
    print(np.cumsum(pca.explained_variance_ratio_))
    print()
