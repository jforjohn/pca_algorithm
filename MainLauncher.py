##
from MyKmeans import MyKmeans
from MyKmedoids import MyKmedoids
from MyFuzzyCmeans import MyFuzzyCmeans
from MyPreprocessing import MyPreprocessing
from sklearn.cluster import AgglomerativeClustering
from scipy.io.arff import loadarff
import pandas as pd
from config_loader import load, clf_names
import argparse
import sys
from time import time

##
from Validation import validation_metrics
from Validation import best_k

##
if __name__ == '__main__':
    ##
    # 1: Kmeans, 2: Kmedoids, 3: Fuzzy C-means
    #accepted_algorithms = [1,2,3]

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
    k = int(config.get('clustering', 'k'))
    tol = float(config.get('clustering', 'tol'))
    max_rep = int(config.get('clustering', 'max_rep'))
    fuzzy_m = int(config.get('clustering', 'fuzzy_m'))
    kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    run = config.get('clustering', 'run')
    clf_options = {
        '1': MyKmeans(k, tol, max_rep),
        '2': MyKmeans(k, tol, max_rep, kmeans_init_type),
        '3': MyKmedoids(k, tol, max_rep),
        '4': MyFuzzyCmeans(k, tol, max_rep, fuzzy_m),
        '5': AgglomerativeClustering(n_clusters=k, linkage='single'),
        '6': AgglomerativeClustering(n_clusters=k, linkage='average'),
        '7': AgglomerativeClustering(n_clusters=k, linkage='complete')
    }

    ## Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_


