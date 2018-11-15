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
from Pair_Plot import Pair_Plot
import matplotlib.pyplot as plt


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
    n_components = config.get('pca', 'n_components').split('-')
    num_plot_features = int(config.get('pca', 'num_plot_features'))

    ## Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    for n_component in n_components:
        # PCA
        n_component = int(n_component)
        pca = MyPCA(n_components=n_component)
        pca.fit(df)

        if hasattr(pca, 'cov_mat'):
            print('Original Covariance Matrix: ')
            print(pca.cov_mat)
            print()

        if hasattr(pca, 'eigval') and hasattr(pca, 'eigvec'):
            print('Original eigen values and corresponding eigen vectors: ')
            print(pca.eigval)
            print(pca.eigvec)
            print()

        print('K max eigen values and corresponding eigen vectors: ')
        print(pca.explained_variance_)
        print(pca.components_)
        print()

        print('Explained variance for %d components: ' %(n_component))
        print(np.cumsum(pca.explained_variance_ratio_))
        print()

        #specialPairPlot(pca.tranformedData)

        ## Sklearn PCA
        print('PCA sklearn algorithm ')
        pca = PCA(n_components=n_component)
        pca.fit_transform(df)
        print(pca.explained_variance_)
        print(np.cumsum(pca.explained_variance_ratio_))
        print()

        print('IncrementalPCA sklearn  algorithm ')
        pca = IncrementalPCA(n_components=n_component)
        pca.fit_transform(df)
        print(pca.explained_variance_)
        print(np.cumsum(pca.explained_variance_ratio_))
        print()

        Pair_Plot(df, num_plot_features)


