##
from MyKmeans import MyKmeans
from MyPreprocessing import MyPreprocessing
from MyPCA import MyPCA
from scipy.io.arff import loadarff
import pandas as pd
from config_loader import load, clf_names
import argparse
import sys
import seaborn as sns

import numpy as np
from time import time
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

    ## Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    # PCA
    clf = MyPCA(2)
    clf.fit(df)
    explained_variance = clf.n_eigval/sum(clf.eigval)

    ##
    pairplot = Pair_Plot(df, 3)
    print()