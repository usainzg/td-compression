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