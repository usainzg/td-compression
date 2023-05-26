from __future__ import division
import copy
import torch
import tltorch
import tensorly as tl
import numpy as np
from scipy.optimize import minimize_scalar

from utils import factorizations


def factorize_layer(
    module,
    factorization='tucker',
    rank=None,
    decompose_weights=False,
    vbmf=0,
    implementation='reconstructed'
):
    init_std = None if decompose_weights else 0.01
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
    # implementation see: https://github.com/tensorly/torch/blob/d27d58f16101b7ecc431372eb218ceda59d8b043/tltorch/functional/convolution.py#L286
    
    if rank is None and vbmf == 0 and factorization != 'tucker':
        raise ValueError('rank must be specified for non-tucker factorization')
    
    if not decompose_weights:
        vbmf = 0 

    if type(module) == torch.nn.modules.conv.Conv2d:
        # rank selection
        if vbmf ==  1:
            ranks = factorizations.estimate_ranks(module)
        elif rank is not None:
            ranks = rank
        else:
            weights = module.weight.data
            ranks = [weights.shape[0]//3, weights.shape[1]//3, weights.shape[2], weights.shape[3]]
        
        # factorize from conv layer
        fact_module = tltorch.FactorizedConv.from_conv(
            module,
            rank=ranks,
            decompose_weights=decompose_weights,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs
        )
    elif type(module) == torch.nn.modules.linear.Linear:
        fact_module = tltorch.FactorizedLinear.from_linear(
            module,
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
        #print('Initializing with std')
        fact_module.weight.normal_(0, init_std)
    
    return fact_module

def factorize_network(
    model_name,
    model,
    tn_decomp,
    rank,
    decompose_weights,
    implementation='reconstructed',
    vbmf=0,
    layers=[],
    exclude=[],
    verbose=False
):
    if model_name.startswith('resnet'):
        # layers to tensorize
        if model_name == "resnet18":
            layer_names = ['layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2', 'layer2.0.conv1', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2', 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2']
        elif model_name == "resnet50":
            layer_names = [
                "layer1.0.conv1",
                "layer1.0.conv2",
                "layer1.0.conv3",
                "layer1.1.conv1",
                "layer1.1.conv2",
                "layer1.1.conv3",
                "layer2.0.conv1",
                "layer2.0.conv2",
                "layer2.0.conv3",
                "layer2.1.conv1",
                "layer2.1.conv2",
                "layer2.1.conv3",
                "layer3.0.conv1",
                "layer3.0.conv2",
                "layer3.0.conv3",
                "layer3.1.conv1",
                "layer3.1.conv2",
                "layer3.1.conv3",
                "layer4.0.conv1",
                "layer4.0.conv2",
                "layer4.0.conv3",
                "layer4.1.conv1",
                "layer4.1.conv2",
                "layer4.1.conv3",
                "layer4.2.conv1",
                "layer4.2.conv2",
                "layer4.2.conv3",
            ]
        elif model_name == "resnet34":
            layer_names = [
                "layer1.0.conv1",
                "layer1.0.conv2",
                "layer1.1.conv1",
                "layer1.1.conv2",
                "layer1.2.conv1",
                "layer1.2.conv2",
                "layer2.0.conv1",
                "layer2.0.conv2",
                "layer2.1.conv1",
                "layer2.1.conv2",
                "layer2.2.conv1",
                "layer2.2.conv2",
                "layer2.3.conv1",
                "layer2.3.conv2",
                "layer3.0.conv1",
                "layer3.0.conv2",
                "layer3.1.conv1",
                "layer3.1.conv2",
                "layer3.2.conv1",
                "layer3.2.conv2",
                "layer3.3.conv1",
                "layer3.3.conv2",
                "layer3.4.conv1",
                "layer3.4.conv2",
                "layer3.5.conv1",
                "layer3.5.conv2",
                "layer4.0.conv1",
                "layer4.0.conv2",
                "layer4.1.conv1",
                "layer4.1.conv2",
                "layer4.2.conv1",
                "layer4.2.conv2",
            ]
        else:
            raise NotImplementedError('not recognized resnet model')

        fact_model = copy.deepcopy(model)
        # factorize resnet
        for i, (name, module) in enumerate(model.named_modules()):
            if name in layer_names:
                if verbose:
                    print(f'factorizing: {name}')
                fact_module = factorize_layer(
                    module=module, 
                    factorization=tn_decomp, 
                    rank=rank, 
                    vbmf=vbmf,
                    decompose_weights=decompose_weights,
                    implementation=implementation
                )
                layer, block, conv = name.split('.')
                conv_to_replace = getattr(getattr(fact_model, layer), block)
                setattr(conv_to_replace, conv, fact_module)
    
    elif model_name.startswith('vgg'):
        fact_model = copy.deepcopy(model)
        # factorize vgg
        for i, (name, module) in enumerate(model.named_modules()):
            if type(module) == torch.nn.modules.conv.Conv2d:
                if name == 'features.0':
                    continue # Skip first layer
                if verbose:
                    print(f'factorizing: {name}')
                fact_layer = factorize_layer(
                    module=module, 
                    factorization=tn_decomp, 
                    rank=rank, 
                    vbmf=vbmf,
                    decompose_weights=decompose_weights,
                    implementation=implementation
                )
                layer, block = name.split('.')
                conv_to_replace = getattr(fact_model, layer)
                setattr(conv_to_replace, block, fact_layer)
    
    else:
        raise NotImplementedError(model_name)

    return fact_model

    
