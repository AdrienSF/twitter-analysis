
import numpy as np
import cupy as cp
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

from helpers import get_phi

tl.set_backend('cupy')


print("Libraries Imported")


## Create Simple Tensor


k = 5
alpha_0 = 0.1
a, phi, _, _ = get_phi(k, 1000, 10000, 10,alpha_val=alpha_0/k)

## Verify WM2W = I
M1 = tl.mean(a, axis=0)
a_cent = (a - M1)
true_res = tl.eye(k)

p = PCA(k, alpha_0, 5000)
p.fit(a_cent)
#M2 = (alpha_0 + 1)*tl.mean(batched_tensor_dot(a_cent, a_cent), axis=0)
#W = p.projection_weights_ / tl.sqrt(p.whitening_weights_)[None, :]
#res = tl.dot(tl.dot(W.T, M2), W)
#assert_array_almost_equal(res, true_res)
#print("Second Moment Complete")

## Verify L close to 0
learning_rate = 0.01

t = TLDA(k,n_senti=1, alpha_0= alpha_0, n_iter_train=10000,
        n_iter_test=150, batch_size=5000, # increase train, 2000
            learning_rate=learning_rate,theta=0.5)
print("TLDA Initiated")
TLDAseed = cp.copy(t.factors_)
print("Begin Tensor Decompostion")
a_whit = p.transform(a_cent)
t.fit(a_whit, verbose=True)
print("Tensor Decompostion Complete")

## Verify dL/dv is close to 0

## Verify v^3 = T^3
# M3 = (alpha_0 + 1)*(alpha_0 + 2)/2*tl.mean(batched_tensor_dot(batched_tensor_dot(a_cent, a_cent),a_cent),axis=0)
#print(M3)
# M3_whiten = (alpha_0 + 1)*(alpha_0 + 2)/2*tl.mean(batched_tensor_dot(batched_tensor_dot(a_whit, a_whit),a_whit),axis=0)
# assert_array_almost_equal(tl.tensor(t.factors_), M3_whiten)
# print("Third Moment Complete")

stdycros3 = cp.std(batched_tensor_dot(batched_tensor_dot(a_whit, a_whit),a_whit))
# print('std(y(cross)^3):', stdycros3)
# print('std(a_whit):', cp.std(a_whit))
# print('std(TLDAseed):', cp.std(TLDAseed))
# print('std input to generator:', 0.8770580193070291)
# stdM3_whitencros3 = cp.std(batched_tensor_dot(batched_tensor_dot(M3_whiten, M3_whiten),M3_whiten))
# print('std(M3_whiten(cross)^3):', stdM3_whitencros3)
# stdTLDAseedcros3 = cp.std(batched_tensor_dot(batched_tensor_dot(TLDAseed, TLDAseed),TLDAseed))
# print('std(TLDAseed(cross)^3):', stdTLDAseedcros3)

print("Initial size of factors: " + str(t.factors_.shape))
t.factors_ = p.reverse_transform(t.factors_)
print("Post Unwhitening size of factors: " + str(t.factors_.shape))
t.predict(a_cent,w_mat=False,doc_predict=False)
print("Post Processing size of factors: " + str(t.factors_.shape))

factors    = t.factors_
print(cp.sum(t.factors_,axis=0))
print(cp.sum(t.factors_,axis=1))
nphi = np.array(phi)
print(nphi.shape)
print('argsort sum(a, axis=0):',np.argsort(np.sum(a, axis=0))[-10:])
tophi = np.argsort(nphi[0,0])[-10:]
print('argsort phi:',tophi)
print(nphi[0,0][tophi])

topfactors = np.argsort(factors[:,0])[-10:]
print('argsort factors',topfactors)
print(factors[:,0][topfactors])

diff = len(set(tophi.tolist()) - set(topfactors.tolist()))
print('diff:', diff)



assert_array_almost_equal(phi, factors)

