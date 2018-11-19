##
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##

def Grid_Plot(df, n_col= 4):
    sns.set()
    random_select = np.random.choice(len(df.columns), n_col, replace=False)

    # create a plot in a grid with different type of graphs
    grid = sns.PairGrid(df[df.columns[random_select]])

    grid = grid.map_upper(plt.scatter, color='darkred')

    grid = grid.map_diag(plt.hist, bins=10, color='darkred', edgecolor='k')

    grid = grid.map_lower(sns.kdeplot, cmap='Reds')
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def pairPlot_grid(df, labels, n_col= 4):
    random_select = np.random.choice(len(df.columns), n_col, replace=False)
    sns.set(style="ticks", color_codes=True, palette='Set2')
    df_labels = pd.DataFrame(list(map(str, labels)), columns=['labels'])
    #df_labels = df_labels.astype(str)
    df_total = pd.concat([df, df_labels], axis=1, sort=False)
    pairplot = sns.pairplot(df_total)#, vars=df.columns[random_select], hue='labels')
    plt.show()

def corr_matrix(df):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(bottom=0.1)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def pair_plot(transform_data, labels, pred_nonpca, pred_pca):
    df_labels = pd.DataFrame(labels, columns=["labels"])
    df_pred_nonpca = pd.DataFrame(pred_nonpca, columns= ["pred_nonpca"])
    df_pred_pca = pd.DataFrame(pred_pca, columns= ["pred_pca"])
    df_transfdata = pd.DataFrame(transform_data, columns=["F1","F2"])

    df_true_labels = pd.concat([df_transfdata, df_labels], axis=1)
    df_nonpca = pd.concat([df_transfdata, df_pred_nonpca], axis=1)
    df_pca = pd.concat([df_transfdata, df_pred_pca], axis=1)

    # three plots sharing Y axis
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.scatter(df_true_labels.F1, df_true_labels.F2, c=df_true_labels.labels)
    ax1.set_title('True labels')
    ax2.scatter(df_nonpca.F1, df_nonpca.F2, c=df_nonpca.pred_nonpca)
    ax2.set_title('Predictions \nKMeans')
    ax3.scatter(df_pca.F1, df_pca.F2, c=df_pca.pred_pca)
    ax3.set_title('Predictions \nKMeans with PCA')
    plt.show()


def specialGridPlot(df, reconstructedData):
    """
    indexes of most correlated features:
    nursery: [1,2,4,7]
    pen-based: [5,11,13,15]
    segment: [9,11,13,16]
    """
    sns.set()
    columns=[1,2,4,7]

    # Reconstructed data frame with original features
    df_reconstructed = pd.DataFrame(reconstructedData, columns=df.columns)

    # create a plot in a grid with different type of graphs
    grid = sns.PairGrid(df_reconstructed[df_reconstructed.columns[columns]])

    grid = grid.map_upper(plt.scatter, color='darkred')

    grid = grid.map_diag(plt.hist, bins=10, color='darkred', edgecolor='k')

    grid = grid.map_lower(sns.kdeplot, cmap='Reds')
    plt.subplots_adjust(bottom=0.1)

    plt.show()

##
""""
##
specialGridPlot(df, mypca.reconstructedData)

##
pair_plot(mypca.transformedData, labels, labels_nonpca, labels_pca)


##
# pair plot for df with predictions (labels)
n_col=2
random_select = np.random.choice(len(df.columns), n_col, replace=False)

df_labels = pd.DataFrame(labels, columns=["labels"])

df_labeled = pd.concat([df[df.columns[random_select]], df_labels], axis=1)
df_labeled.labels = df_labeled.labels.astype(str)

sns.pairplot(df_labeled, vars=df.columns[random_select],  hue="labels")



##
grid = sns.PairGrid(df_labeled[["a6","a1"]], hue="df_labeled.labels")
#grid.map(plt.scatter)

##

sns.pairplot(df_labeled, vars=["a6","a1"])

"""