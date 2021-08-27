
import numpy as np
import scipy
from scipy.stats import gamma
from sklearn.decomposition import IncrementalPCA

# Import TensorLy
import tensorly as tl
from tensorly.tenalg import kronecker
from tensorly import norm
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac
from tensorly.tenalg.core_tenalg.tensor_product import batched_tensor_dot
from tensorly.testing import assert_array_equal, assert_array_almost_equal

# Import utility functions from other files
from pca_cupy import PCA
from tlda_final_cupy import TLDA

tl.set_backend('cupy')


print("Libraries Imported")


## Create Simple Tensor
a = tl.tensor([[2, 1, 0, 1, 5],
                [1, 0, 3, 2, 3],
                [0, 0, 4, 1, 1],
                [1, 1, 4, 4, 1],
                [1, 1, 10, 3, 1],
                [4, 0, 1, 4, 0],
                [1,3, 3, 2, 1]])
alpha_0 = 0.3
## Verify WM2W = I
M1 = tl.mean(a, axis=0)
a_cent = (a - M1)
k = 3
true_res = tl.eye(k)

p = PCA(k, alpha_0, 4)
p.fit(a_cent)
M2 = (alpha_0 + 1)*tl.mean(batched_tensor_dot(a_cent, a_cent), axis=0)
W = p.projection_weights_ / tl.sqrt(p.whitening_weights_)[None, :]
res = tl.dot(tl.dot(W.T, M2), W)
assert_array_almost_equal(res, true_res)
print("Second Moment Complete")

## Verify L close to 0
learning_rate = 0.01

t = TLDA(k,n_senti=1, alpha_0= alpha_0, n_iter_train=10000,n_iter_test=150, batch_size=100, # increase train, 2000
            learning_rate=learning_rate)
print("TLDA Initiated")
print("Begin Tensor Decompostion")
a_whit = p.transform(a_cent)
t.fit(a_whit, verbose=True)
print("Tensor Decompostion Complete")

## Verify dL/dv is close to 0

## Verify v^3 = T^3
M3 = (alpha_0 + 1)*(alpha_0 + 2)/2*tl.mean(batched_tensor_dot(batched_tensor_dot(a_cent, a_cent),a_cent),axis=0)
#print(M3)
M3_whiten = (alpha_0 + 1)*(alpha_0 + 2)/2*tl.mean(batched_tensor_dot(batched_tensor_dot(a_whit, a_whit),a_whit),axis=0)
assert_array_almost_equal(tl.tensor(t.factors_), M3_whiten)
print("Third Moment Complete")