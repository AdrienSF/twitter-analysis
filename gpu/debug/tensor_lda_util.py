import numpy as np

from scipy.special import comb, digamma, gammaln
from scipy.stats import gamma
# import sparse ????
import scipy

# Import TensorLy
import tensorly as tl
from tensorly import norm
from tensorly.tenalg import kronecker
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.decomposition import symmetric_parafac_power_iteration
tl.set_backend('cupy')
device = 'gpu'#cuda

def get_ei(length, i):
    '''Get the ith standard basis vector of a given length'''
    e = tl.zeros(length)
    e[i] = 1
    return e

def dirichlet_expectation(alpha):
    '''Normalize alpha using the dirichlet distribution'''
    return digamma(alpha) - digamma(sum(alpha))

def smooth_beta(beta, smoothing = 0.01):
    '''Smooth the existing beta so that it all positive (no 0 elements)'''
    smoothed_beta = beta * (1 - smoothing)
    smoothed_beta += (np.ones((beta.shape[0], beta.shape[1])) * (smoothing/beta.shape[0]))

    assert np.all(abs(np.sum(smoothed_beta, axis=0) - 1) <= 1e-6), 'sum not close to 1'
    assert smoothing <= 1e-4 or np.all(smoothed_beta > 1e-10), 'zero values'
    return smoothed_beta

def simplex_proj(V):
    '''Project V onto a simplex'''
    v_len = V.size
    U = np.sort(V)[::-1]
    cums = np.cumsum(U, dtype=float) - 1
    index = np.reciprocal(np.arange(1, v_len+1, dtype=float))
    inter_vec = cums * index
    to_befind_max = U - inter_vec
    max_idx = 0

    for i in range(0, v_len):
        if (to_befind_max[v_len-i-1] > 0):
            max_idx = v_len-i-1
            break
    theta = inter_vec[max_idx]
    p_norm = V - theta
    p_norm[p_norm < 0.0] = 0.0
    return (p_norm, theta)

def non_negative_adjustment(M):
    '''Adjust M so that it is not negative by projecting it onto a simplex'''
    M_on_simplex = np.zeros(M.shape)
    M = tl.to_numpy(M)

    for i in range(0, M.shape[1]):
        projected_vector, theta = simplex_proj(M[:, i] - np.amin(M[:, i]))
        projected_vector_revsign, theta_revsign = simplex_proj(-1*M[:, i] - np.amin(-1*M[:, i]))

        if (theta < theta_revsign):
            M_on_simplex[:, i] = projected_vector
        else:
            M_on_simplex[:, i] = projected_vector_revsign
    return M_on_simplex

def perplexity (documents, beta, alpha, gamma):
    '''get perplexity of model, given word count matrix (documents)
    topic/word distribution (beta), weights (alpha), and document/topic
    distribution (gamma)'''

    elogbeta = np.log(beta)

    corpus_part = np.zeros(documents.shape[0])
    for i, doc in enumerate(documents):
        doc_bound = 0.0
        gammad = gamma[i]
        elogthetad = dirichlet_expectation(gammad)

        for idx in np.nonzero(doc)[0]:
            doc_bound += doc[idx] * log_sum_exp(elogthetad + elogbeta[idx].T)

        doc_bound += np.sum((alpha - gammad) * elogthetad)
        doc_bound += np.sum(gammaln(gammad) - gammaln(alpha))
        doc_bound += gammaln(np.sum(alpha)) - gammaln(np.sum(gammad))

        corpus_part[i] = doc_bound

    #sum the log likelihood of all the documents to get total log likelihood
    log_likelihood = np.sum(corpus_part)
    total_words = np.sum(documents)

    #perplexity is - log likelihood / total number of words in corpus
    return (-1*log_likelihood / total_words)

def log_sum_exp(x):
    '''calculate log(sum(exp(x)))'''
    a = np.amax(x)
    return a + np.log(np.sum(np.exp(x - a)))

def get_M1(x):
    '''Get M1 moment by averaging all document vectors ((1) from [1])
    Parameters
    ----------
    x : ndarray
    Returns
    -------
    M1 : tensor of shape (1, x.shape[1])
    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    return tl.mean(x, 0)

def get_M2(x, M1, alpha_0):
    '''Get M2 moment using (2) from [1]
    Parameters
    ----------
    x : ndarray
    M1 : tensor of shape (1, x.shape[1]) equal to M1 moment (1) from [1]
    alpha_0 : float equal to alpha_0 from [1]
    Returns
    -------
    M2 : tensor of shape (x.shape[1], x.shape[1]) equal to (2) from [1]
    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    sum_ = batched_tensor_dot(x, x)
    sum_ = tl.mean(sum_, axis=0)
    sum_ = sum_ - tl.diag(tl.mean(x, axis=0))
    sum_ *= (alpha_0 + 1)
    sum_ = sum_ - alpha_0*tl.reshape(kronecker([M1, M1]), sum_.shape)
    return sum_

def whiten(M, k, condition = False):
    '''Get W and W^(-1), where W is the whitening matrix for M, using the rank-k svd
    Parameters
    ----------
    M : tensor of shape (vocabulary_size, vocabulary_size) equal to
        the M2 moment tensor
    k : integer equal to the number of topics
    condition : bool, optional
        if True, print the M2 condition number
    Returns
    -------
    W : tensor of shape (vocabulary_size, number_topics) equal to the whitening
        tensor for M
    W_inv : tensor of shape (number_topics, vocabulary_size) equal to the inverse
        of W
    '''
    U, S, V = tl.partial_svd(M, n_eigenvecs=k)
    W_inv = tl.dot(U, tl.diag(tl.sqrt(S)))

    if condition == True:
        print("M2 condition number: " + str(np.amax(S)/np.amin(S)))

    return (U / tl.sqrt(S)[None, :]), W_inv

def get_M3 (x, M1, alpha_0):
    '''Get M3 moment using (3) from [1]
    Parameters
    ----------
    x : ndarray
    M1 : tensor of shape (1, x.shape[1]) equal to M1 moment (1) from [1]
    alpha_0 : float equal to alpha_0 from [1]
    Returns
    -------
    M3 : tensor of shape (x.shape[1], x.shape[1], x.shape[1]) equal to (3) from [1]
    
    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    ns = x.shape[0]
    n = x.shape[1]

    sum_ = x
    for _ in range(2):
        #print(sum_)
        sum_ = batched_tensor_dot(sum_, x)
    sum_ = tl.sum(sum_, axis=0)

    #issue: no dense equivalent
    diag = tl.zeros((ns, n, n))
    for i in range(ns):
        diag[i] = tl.diag(x[i])
    sum_ -= tl.sum(batched_tensor_dot(diag, x), axis=0)
    sum_ -= tl.sum(batched_tensor_dot(x, diag), axis=0)

    eye_mat = tl.eye(n)
    for _ in range(2):
        eye_mat = batched_tensor_dot(eye_mat, tl.eye(n))
    eye_mat = tl.sum(eye_mat, axis=0)
    sum_ += 2*eye_mat*tl.sum(x, axis=0)

    #final sym term
    tot = tl.zeros((1, n * n * n))
    for i in range(n):
        for j in range(n):
            tot += tl.sum(x[:,i]*x[:,j])*kronecker([get_ei(n, i), get_ei(n, j), get_ei(n, i)])
    sum_ -= tl.reshape(tot, (n, n, n))
    sum_ *= (alpha_0 + 1)*(alpha_0+2)/(2*ns)

    M1_mat = tl.tensor([M1,]*n)*tl.sum(x, axis=0)[:, None]
    eye1 = tl.eye(n)
    eye2 = batched_tensor_dot(eye1, eye1)
    tot2 = tl.sum(batched_tensor_dot(eye2, M1_mat), axis=0)+tl.sum(batched_tensor_dot(batched_tensor_dot(eye1, M1_mat), eye1), axis=0)+tl.sum(batched_tensor_dot(M1_mat, eye2), axis=0)
    sum_ -= alpha_0*(alpha_0 + 1)/(2*ns)*tot2

    sum_ += alpha_0*alpha_0*(tl.reshape(kronecker([M1, M1, M1]), (n, n, n)))
    return sum_

def fit_exact(x, num_tops, alpha_0, verbose = True):
    '''Fit the documents to num_tops topics using the method of moments as
    outlined in [1]
    Parameters
    ----------
    x : ndarray of shape (number_documents, vocabulary_size) equal to the word
        counts in each document
    num_tops : int equal to the number of topics to fit x to
    alpha_0 : float equal to alpha_0 from [1]
    verbose : bool, optional
        if True, print the eigenvalues and best scores during the decomposition
    Returns
    -------
    w3_learned : tensor of shape (1, number_topics) equal to the weights
                 (eigenvalues) for each topic
    f3_reshaped : tensor of shape (number_topics, vocabulary_size) equal to the
                  learned topic/word distribution (requires adjustment using
                  the inference method)
    References
    ----------
    [1] Furong Huang, U. N. Niranjan, Mohammad Umar Hakeem, and Animashree Anandkumar,
        2014. Online Tensor Methods for Learning Latent Variable Models. In the
        Journal of Machine Learning Research 2014.
    '''
    M1 = get_M1(x)
    M2_img = get_M2(x, M1, alpha_0)

    W, W_inv = whiten(M2_img, num_tops)
    X_whitened = tl.dot(x, W)
    M1_whitened = tl.dot(M1, W)

    M3_final = get_M3(X_whitened, M1_whitened, alpha_0)
    w3_learned, f3_learned = symmetric_parafac_power_iteration(M3_final, rank=num_tops, n_repeat=15, verbose=verbose)

    f3_reshaped = (tl.dot(f3_learned, W_inv.T))
    return w3_learned, f3_reshaped, M3_final