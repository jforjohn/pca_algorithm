##
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

##

def Pair_Plot(df, n_col= 4):

    random_select = np.random.choice(len(df.columns), n_col, replace=False)

    # create a plot in a grid with different type of graphs
    grid = sns.PairGrid(df[df.columns[random_select]])

    grid = grid.map_upper(plt.scatter, color='darkred')

    grid = grid.map_diag(plt.hist, bins=10, color='darkred', edgecolor='k')

    grid = grid.map_lower(sns.kdeplot, cmap='Reds')
    plt.subplots_adjust(bottom=0.1)

    plt.show()


##
"""
##
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
plt.subplots_adjust(bottom = 0.1)


# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

##
"""
"""
selected_col= ['intensity-mean', 'rawred-mean', 'rawblue-mean', 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean']

sns.pairplot(df[selected_col])
"""