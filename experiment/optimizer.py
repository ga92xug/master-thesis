import hydra
import torch
from torch import nn

import nn as enn
import optimizers_L1L2

# TODO Stefan check if this is correct
def build_optimizer_sfcnn(params, l1, lamb_conv_L1, lamb_conv_L2, 
                          lamb_fully_L1, lamb_fully_L2, lamb_softmax_L2, lamb_bn_L1,
                          lamb_bn_L2,
                          lr,):
    # optimizer as in "Learning Steerable Filters for Rotation Equivariant CNNs"
    # https://arxiv.org/abs/1711.07289
    
    # split up parameters into groups, named_parameters() returns tuples ('name', parameter)
    # each group gets its own regularization gain
    batchnormLayers = [m for m in params.modules() if isinstance(m,
                                                                     (
                                                                    enn.BatchNorm,
                                                                    enn.InducedNormBatchNorm,
                                                                    enn.GroupStandardization,
                                                                    enn.GroupNorm,
                                                                    enn.InducedNormGroupNorm,
                                                                      )
                                                                )]
    linearLayers = [m for m in params.modules() if isinstance(m, nn.Linear)]
    convlayers = [m for m in params.modules() if isinstance(m, (nn.Conv2d, enn.R2Conv))]
    weights_conv = [p for m in convlayers for n, p in m.named_parameters() if n.endswith('weights') or n.endswith("weight")]
    biases = [p for n, p in params.named_parameters() if n.endswith('bias')]
    weights_bn = [p for m in batchnormLayers for n, p in m.named_parameters()
                  if n.endswith('weight') or n.split('.')[-1].startswith('weight')
                  ]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]
    # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    print("SFCNN optimizer")
    for n, p in params.named_parameters():
        if p.requires_grad and not n.endswith(('weight', 'weights', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(n))
    param_groups = [dict(params=weights_conv, lamb_L1=lamb_conv_L1, lamb_L2=lamb_conv_L2, weight_decay=lamb_conv_L2),
                    dict(params=weights_bn, lamb_L1=lamb_bn_L1, lamb_L2=lamb_bn_L2, weight_decay=lamb_bn_L2),
                    dict(params=weights_fully, lamb_L1=lamb_fully_L1, lamb_L2=lamb_fully_L2, weight_decay=lamb_fully_L2),
                    dict(params=weights_softmax, lamb_L1=0, lamb_L2=lamb_softmax_L2, weight_decay=lamb_softmax_L2),
                    dict(params=biases, lamb_L1=0, lamb_L2=0, weight_decay=0)]
    if l1:
        return optimizers_L1L2.Adam(param_groups, lr=lr, betas=(0.9, 0.999))
    else:
        return torch.optim.Adam(param_groups, lr=lr, betas=(0.9, 0.999))


def build_optimizer(model, cfg):
    if cfg.optimizer.name == "sfcnn":
        # optimize as in "Learning Steerable Filters for Rotation Equivariant CNNs"
        # https://arxiv.org/abs/1711.07289
        return build_optimizer_sfcnn(model, cfg)
    elif cfg.optimizer.name == "Adam":
        return torch.optim.Adam(model.parameters(),
                                            lr=cfg.optimizer.lr,
                                            weight_decay=cfg.optimizer.weight_decay
                                            )
    elif cfg.optimizer.name == "SGD":
        return torch.optim.SGD(model.parameters(),
                                            lr=cfg.optimizer.lr,
                                            momentum=cfg.optimizer.momentum,
                                            weight_decay=cfg.optimizer.weight_decay)
    else:
        raise Exception(f"Unknown optimizer {cfg.optimizer.name}")