from __future__ import division
import torch
import tltorch
import tensorly as tl
import numpy as np
from scipy.optimize import minimize_scalar

def factorize_layer(
    module,
    factorization='tucker',
    rank=0.5,
    decompose_weights=True,
    init_std=None
):
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None

    if type(module) == torch.nn.modules.conv.Conv2d:
        fact_module = tltorch.FactorizedConv.from_conv(
            module,
            rank=rank,
            decompose_weights=decompose_weights,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
        )
    elif type(module) == torch.nn.modules.linear.Linear:
        fact_module = tltorch.FactorizedLinear.from_linear(
            module,
            #in_tensorized_features=get_prime_factors(module.in_features),
            #out_tensorized_features=get_prime_factors(module.out_features),
            n_tensorized_modes=3,
            rank=rank,
            factorization=factorization,
            decompose_weights=decompose_weights,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
        )
    else:
        raise NotImplementedError(type(module))
    
    if init_std:
        fact_module.weight.normal_(0, init_std)
    
    return fact_module

def factorize_network(
    model,
    layers=[],
    exclude=[],
    verbose=False
):
    return NotImplementedError

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    diag_0 = EVBMF(unfold_0)[1]
    diag_1 = EVBMF(unfold_1)[1]
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    ranks = [ranks[0], ranks[1], weights.shape[2], weights.shape[3]]
    return ranks

"""
Important note: 
- This code is taken from: https://github.com/CasvandenBogaard/VBMF
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
"""
def EVBMF(Y, sigma2=None, H=None):
    """Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical VBMF. 
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
    
    sigma2 : int or None (default=None)
        Variance of the noise on Y.
        
    H : int or None (default = None)
        Maximum rank of the factorized matrices.
        
    Returns
    -------
    U : numpy-array
        Left-singular vectors. 
        
    S : numpy-array
        Diagonal matrix of singular values.
        
    V : numpy-array
        Right-singular vectors.
        
    post : dictionary
        Dictionary containing the computed posterior values.
        
        
    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
    
    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.     
    """   
    L,M = Y.shape #has to be L<=M

    if H is None:
        H = L

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)
    
    #SVD of the input matrix, max rank of H
    U,s,V = np.linalg.svd(Y)
    U = U[:,:H]
    s = s[:H]
    V = V[:H].T 

    #Calculate residual
    residual = 0.
    if H<L:
        residual = np.sum(np.sum(Y**2)-np.sum(s**2))

    #Estimation of the variance when sigma2 is unspecified
    if sigma2 is None: 
        xubar = (1+tauubar)*(1+alpha/tauubar)
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        upper_bound = (np.sum(s**2)+residual)/(L*M)
        lower_bound = np.max([s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])

        scale = 1.#/lower_bound
        s = s*np.sqrt(scale)
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale

        sigma2_opt = minimize_scalar(EVBsigma2, args=(L,M,s,residual,xubar), bounds=[lower_bound, upper_bound], method='Bounded')
        sigma2 = sigma2_opt.x

    #Threshold gamma term
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    pos = np.sum(s>threshold)

    #Formula (15) from [2]
    d = np.multiply(s[:pos]/2, 1-np.divide((L+M)*sigma2, s[:pos]**2) + np.sqrt((1-np.divide((L+M)*sigma2, s[:pos]**2))**2 -4*L*M*sigma2**2/s[:pos]**4) )

    #Computation of the posterior
    post = {}
    post['ma'] = np.zeros(H) 
    post['mb'] = np.zeros(H)
    post['sa2'] = np.zeros(H) 
    post['sb2'] = np.zeros(H) 
    post['cacb'] = np.zeros(H)  

    tau = np.multiply(d, s[:pos])/(M*sigma2)
    delta = np.multiply(np.sqrt(np.divide(M*d, L*s[:pos])), 1+alpha/tau)

    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos])/(L*M))
    post['sigma2'] = sigma2
    post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 
                     + np.sum(M*np.log(tau+1) + L*np.log(tau/alpha +1) - M*tau))

    return U[:,:pos], np.diag(d), V[:,:pos], post

def EVBsigma2(sigma2,L,M,s,residual,xubar):
    H = len(s)

    alpha = L/M
    x = s**2/(M*sigma2) 

    z1 = x[x>xubar]
    z2 = x[x<=xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum( np.log( np.divide(tau_z1+1, z1)))
    term4 = alpha*np.sum(np.log(tau_z1/alpha+1))
    
    obj = term1+term2+term3+term4+ residual/(M*sigma2) + (L-H)*np.log(sigma2)

    return obj

def tau(x, alpha):
    return 0.5 * (x-(1+alpha) + np.sqrt((x-(1+alpha))**2 - 4*alpha))