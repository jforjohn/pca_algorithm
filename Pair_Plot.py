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


def pair_plot(df):
    df_labels = pd.DataFrame(labels, columns= ["labels"])
    df_pred = pd.DataFrame(clf.labels, columns= ["predictions"])

    df_predictions = pd.concat([df, df_labels, df_pred], axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(5, 5))

    df1.plot(ax=axes[0, 0])
    df2.plot(ax=axes[0, 1])
    axs[0, 0].hist(df_predictions[0])
    axs[0, 1].scatter(data[0], data[1])

    plt.show()


def specialGridPlot(df):
    sns.set()
    columns=[9,11,13,16]

    # Reconstructed data frame with original features
    df_reconstructed = pd.DataFrame(pca.reconstructedData, columns=df.columns)

    # create a plot in a grid with different type of graphs
    grid = sns.PairGrid(df_reconstructed[df_reconstructed.columns[columns]])

    grid = grid.map_upper(plt.scatter, color='darkred')

    grid = grid.map_diag(plt.hist, bins=10, color='darkred', edgecolor='k')

    grid = grid.map_lower(sns.kdeplot, cmap='Reds')
    plt.subplots_adjust(bottom=0.1)

    plt.show()


""""
##
specialGridPlot(df)

##

Grid_Plot(df,3)


##
# values of most correlated features:
# 1,2,4,7 nursery
# 5,11,13,15 pen-based
# 9,11,13,16 segment


##

n_col=2
random_select = np.random.choice(len(df.columns), n_col, replace=False)

df_labels = pd.DataFrame(labels, columns=["labels"])

df_labeled = pd.concat([df[df.columns[random_select]], df_labels], axis=1)
df_labeled.labels = df_labeled.labels.astype(str)
sns.pairplot(df_labeled, vars=["exred-mean","exgreen-mean","value-mean"],  hue="labels")


##
df_labeled = pd.concat([df, pd.DataFrame(labels, columns=["labels"])], axis=1)

##
df_labeled.labels = df_labeled.labels.astype(str)

##
grid = sns.PairGrid(df_labeled[["a1","a2"]], hue="df.labeled.labels")
grid.map(plt.scatter)

"""