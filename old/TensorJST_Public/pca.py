import tensorly as tl
from sklearn.decomposition import IncrementalPCA

class PCA():
    def __init__(self, n_eigenvec, alpha_0, batch_size): # n_eigenvec here corresponds to n_topic in the LDA
        self.n_eigenvec = n_eigenvec
        self.alpha_0 = alpha_0
        self.batch_size = batch_size
        self.pca = IncrementalPCA(n_components = self.n_eigenvec, batch_size = self.batch_size)

    def fit(self, X):
        '''Fit the entire data to get the projection weights (singular vectors) and
        whitening weights (scaled explained variance) of a centered input dataset X.
        Parameters
        ----------
        X : tensor containing all input documents
        '''
        self.pca.fit(X*tl.sqrt(self.alpha_0+1))
        self.projection_weights_ = tl.transpose(self.pca.components_)
        self.whitening_weights_ = self.pca.explained_variance_*(X.shape[0] - 1)/(X.shape[0])

    def partial_fit(self, X_batch):
        '''Fit a batch of data and update the projection weights (singular vectors) and
        whitening weights (scaled explained variance) accordingly using a centered
        batch of the input dataset X.
        Parameters
        ----------
        X_batch : tensor containing a batch of input documents
        '''
        self.pca.partial_fit(X_batch)
        self.projection_weights_ = tl.transpose(self.pca.components_)
        self.whitening_weights_ = self.pca.explained_variance_*(X.shape[0] - 1)/(X.shape[0])

    def transform(self, X):
        '''Whiten some centered tensor X using the fitted PCA model.
        Parameters
        ----------
        X : centered input tensor
        '''
        return tl.dot(X, (self.projection_weights_ / tl.sqrt(self.whitening_weights_)[None, :]))

    def reverse_transform(self, X):
        '''Unwhiten some whitened tensor X using the fitted PCA model.
        Parameters
        ----------
        X : whitened input tensor
        '''
        return tl.dot(X, (self.projection_weights_ * tl.sqrt(self.whitening_weights_)).T)