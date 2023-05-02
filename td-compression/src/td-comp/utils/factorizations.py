import torch
import tltorch as tl
from VBMF import VBMF

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
        fact_module = tl.FactorizedConv.from_conv(
            module,
            rank=rank,
            decompose_weights=decompose_weights,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs
        )
    elif type(module) == torch.nn.modules.linear.Linear:
        fact_module = tl.FactorizedLinear.from_linear(
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
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks