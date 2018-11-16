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

def kmeans_comparison(k,tol,max_rep, data, name):
    start = time()
    clf = MyKmeans(k, tol, max_rep)
    clf.fit(data)
    duration = time() - start
    metrics = validation_metrics(df, labels, clf.labels_)
    max_rep = 100 - clf.max_rep

    metrics.update({"Time": duration,
                    "Repetitions": max_rep})
    validations = pd.DataFrame.from_dict(metrics, orient='index',
                                            columns=[clf.name+name])
    return validations


##
if __name__ == '__main__':
    ##
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
    #kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    n_components = int(config.get('pca', 'n_components')) #.split('-')
    num_plot_features = int(config.get('pca', 'num_plot_features'))
    k = int(config.get('clustering', 'k'))
    tol = float(config.get('clustering', 'tol'))
    max_rep = int(config.get('clustering', 'max_rep'))

    ## Preprocessing
    preprocess = MyPreprocessing()
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    #for n_component in n_components:

    # PCA

    ## MyPCA
    mypca = MyPCA(n_components=n_components)
    mypca.fit(df)

    print('Original Covariance Matrix: ')
    print(mypca.cov_mat)
    print()

    print('Original eigen values and corresponding eigen vectors: ')
    print(mypca.eigval)
    print(mypca.eigvec)
    print()

    print('K max eigen values and corresponding eigen vectors: ')
    print(mypca.explained_variance_)
    print(mypca.components_)
    print()

    print('Explained variance for %d components: ' %(n_components))
    print(np.cumsum(mypca.explained_variance_ratio_))
    print()

    #specialPairPlot(pca.tranformedData)

    ## Sklearn PCA
    print('PCA sklearn algorithm ')
    pca_skl = PCA(n_components=n_components)
    pca_skl.fit_transform(df)
    print(pca_skl.explained_variance_)
    print(np.cumsum(pca_skl.explained_variance_ratio_))
    print()

    print('IncrementalPCA sklearn  algorithm ')
    ipca_skl = IncrementalPCA(n_components=n_components)
    ipca_skl.fit_transform(df)
    print(ipca_skl.explained_variance_)
    print(np.cumsum(ipca_skl.explained_variance_ratio_))
    print()

    Pair_Plot(df, num_plot_features)
    
    diff_explained = list(map(np.cumsum, [mypca.explained_variance_,pca_skl.explained_variance_,ipca_skl.explained_variance_]))
    df_plot = pd.DataFrame(np.array(
        diff_explained).T,
        columns=['mypca','pca_skl', 'ipca_skl'])
    df_plot.plot.bar(rot=0)
    

    
    ## Kmeans comparison
    validations_nonpca = kmeans_comparison(k, tol, max_rep, df, '')
    validations_pca = kmeans_comparison(k, tol, max_rep, mypca.transformedData, 'PCA')
    
    validations = pd.concat([validations_nonpca, validations_pca], axis=1)
    print(validations)

    # Show the percentage of repetitions
    validations.iloc[-1, :] = validations.iloc[-1, :]/100.0

    #validations = validations.drop(validations.index[len(validations.index) - 1], axis=0)
    validations.plot.bar(rot=30)
    plt.subplots_adjust(bottom=0.2)
    plt.show()