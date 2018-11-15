import numpy as np
import pandas as pd

class MyPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        if self.n_components >= data.shape[1]:
            raise Exception('It should n_components < number of features (columns of data)')

        # Step 3: Compute the d-dimensional mean vector (i.e., the means of every dimension of the
        # whole data set)
        mean_vector = np.mean(data, axis=0)
        data_adjust = data - mean_vector

        # Step4: Compute the covariance matrix of the whole data set.
        #cov_mat = np.cov(data.T)
        #print(cov_mat)
        self.cov_mat = np.matmul(data_adjust.T, data_adjust) / (data_adjust.shape[0]-1)

        # Step 5. Calculate eigenvectors (e1, e2, …, ed) and their corresponding eigenvalues of the
        # covariance matrix.
        self.eigval, self.eigvec = np.linalg.eig(self.cov_mat)

        # Step 6. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the
        # largest eigenvalues to form a new d x k dimensional matrix.
        choose_n =  sorted(zip(self.eigval, self.eigvec.T), reverse=True)[:self.n_components]
        self.n_eigval = np.array(list(map(lambda x: x[0], choose_n)))
        self.n_eigvec = np.array(list(map(lambda x: x[1], choose_n))).T
        
        # Step 7. Derive the new data set. Use this d x k eigenvector matrix to transform the samples
        # onto the new subspace.
        self.transformedData = np.matmul(self.n_eigvec.T, data_adjust.T).T

        # Step 9. Reconstruct the data set back to the original one
        self.data_new = np.matmul(self.transformedData, self.n_eigvec.T) + mean_vector

'''
data = np.array([[2,3,8],
                 [3,5,12],
                 [1,4,42],
                 [10,12,0],
                 [11,13, 8],
                 [12,10,10]])
clf = MyPCA(n_components=2)
clf.fit(data)
'''