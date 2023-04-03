import torch

from group_theory import kernels
import nn

import numpy as np
np.set_printoptions(precision=3, linewidth=10000, suppress=True)

r2_act = nn.rot2dOnR2(N=4)
feat_type_in = nn.FieldType(r2_act, [r2_act.trivial_repr])
feat_type_out = nn.FieldType(r2_act, [r2_act.regular_repr])
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=3)