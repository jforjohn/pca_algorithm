
##
import matplotlib.pyplot as plt

##
def validation_metrics(df, y_true, y_pred, k_max= 15):
    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import silhouette_score

    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import adjusted_rand_score

    DB= davies_bouldin_score(df, y_pred)
    SC= silhouette_score(df, y_pred)
    AMI= adjusted_mutual_info_score(y_true, y_pred)
    RS= adjusted_rand_score(y_true, y_pred)

    return {'DaviesDublin': DB, 'SilhouetteSc': SC, 'AdjustedMutualInfo': AMI, 'AdjustedRandSc': RS}

##
#metric= validation_metrics(X, y_true, y_pred=pred)


##

def best_k(df, config_file, k_max = 15):
    """
    Models:
            '1': 'KMeans',
            '2': 'KMeans++',
            '3': 'KMedoids',
            '4': 'FuzzyCMeans',
            '5': 'AggloSingle',
            '6': 'AggloAverage',
            '7': 'AggloComplete'

    """
    from sklearn.cluster import KMeans
    from config_loader import load, clf_names
    from sklearn.metrics import silhouette_score
    from MyKmeans import MyKmeans

    config = load(config_file)
    tol = float(config.get('clustering', 'tol'))
    max_rep = int(config.get('clustering', 'max_rep'))
    kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    x = [1]
    sil = [0]
    for k in range(2, k_max + 1):
        clf = MyKmeans(k, tol, max_rep)
        clf.fit(df)
        pred = clf.labels_
        x += [k]
        sil += [silhouette_score(df, pred, metric='euclidean')]

    plt.figure()
    plt.plot(x, sil, color='green', marker='o')
    plt.title('Silhouette Score ' + str(clf.name))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.ylim((0, 1))
    plt.xlim((1, k_max+1))

    return plt
##

#best_k(2)