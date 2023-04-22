import warnings
from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('../scaling-laws-ecnn') # add parent directory

import numpy as np

from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    SequentialModule,
    GroupTensor,
)
from networks import (
    Restriction,
    EquivariantPool,
    EquivariantConvBlock,
    EquivariantWideConvBlock,
    # WideResNet,
)


class EquivariantWideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 4,
        group: str = "cyclic",
        rotation: int = 4,
        fix_params: bool = False,
        restrict: str = None,  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        layout: Tuple[int] = (16, 16, 32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: Tuple[int] = (None, None, None, None),
        num_classes: int = 10,
    ):
        self.depth = depth
        self.widen_factor = widen_factor
        self.group = group
        self.rotation = rotation
        self.fix_params = fix_params
        self.restrict = restrict
        self.input_channels = input_channels
        self.layout = layout
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_groups = num_groups
        self.num_classes = num_classes

        super(EquivariantWideResNet, self).__init__()
        assert (self.depth - 4) % 6 == 0, "WideResNet depth should be 6n+4."
        n = (self.depth - 4) / 6
        k = self.widen_factor

        if self.fix_params:
            self.wrn = WideResNet(
                depth=self.depth,
                num_classes=self.num_classes,
                widen_factor=self.widen_factor,
                input_channels=self.input_channels,
                layout=self.layout,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )

        print("Wide-Resnet %dx%d" % (self.depth, k))

        # Get group spaces for specified rotations and flips
        if self.group == "cyclic":
            self.gspace = rot2dOnR2(self.rotation)
        elif self.group == "dihedral":
            self.gspace = flipRot2dOnR2(self.rotation)
        elif self.group == "orthogonal":
            self.gspace = flipRot2dOnR2(-1)
        else:
            raise ValueError(
                f'Group "{self.group}" is not know. Available groups: [cyclic, dihedral, orthogonal]'
            )

        # Fix number of parameters for all groups
        self.num_channels = np.array(self.layout)
        
        
        if self.fix_params:
            self.num_channels = calculate_fixed_params(self.num_channels, self.kernel_size, 
                                                   self.group, self.gspace, self.rotation, self.restrict)

        # Add width
        self.num_channels *= np.array([1, k, k, k])

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.input_channels
        )

        # "Lifting" conv from trivial to regular feature fields
        self.conv1 = EquivariantConvBlock(
            in_type=self.input_field_type,
            out_channels=int(self.num_channels[0]),
            frequency=self.rotation,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_groups=self.num_groups[0],
        )

        if self.fix_params:
            self.conv1 = self.iter_fix_param(0, self.conv1, self.wrn.conv1)
            

        self.field_type = self.conv1.out_type

        self.layer1 = self._wide_layer(
            EquivariantWideConvBlock,
            self.num_channels[1],
            n,
            stride=1,
            frequency=self.rotation,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_groups=self.num_groups[1],
        )
        if self.fix_params:
            self.layer1 = self.iter_fix_param(1, self.layer1, self.wrn.layer1, n=n, stride=1)

        self.layer2 = self._wide_layer(
            EquivariantWideConvBlock,
            self.num_channels[2],
            n,
            stride=2,
            frequency=self.rotation,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_groups=self.num_groups[2],
        )
        if self.fix_params:
            self.layer2 = self.iter_fix_param(2, self.layer2, self.wrn.layer2, n=n, stride=2)

        # Restrict last conv and res layers
        self.restrict = Restriction(self.layer2.out_type, self.group, self.rotation, self.restrict)
        self.field_type = self.restrict.out_type

        self.layer3 = self._wide_layer(
            block=EquivariantWideConvBlock,
            out_channels=self.num_channels[3],
            num_blocks=n,
            stride=2,
            frequency=self.rotation,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_groups=self.num_groups[3],
        )
        if self.fix_params:
            self.layer3 = self.iter_fix_param(3, self.layer3, self.wrn.layer3, n=n, stride=2)

        self.invariant_map = EquivariantPool(self.layer3.out_type, invariant_map=True)
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * 2 * 2, self.num_classes
        )

        if self.fix_params:
            # size of wrn total and size of equivariant part
            norm_para = sum([p.numel() for p in self.wrn.parameters() if p.requires_grad])
            del self.wrn
            equi_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
            current_ratio = equi_param / norm_para
            print(f"Equivariant_WRN / WRN parameter ratio: {current_ratio:.3f}")
            

    def _wide_layer(
        self,
        block,
        out_channels: int,
        num_blocks: int,
        stride: int,
        frequency: int,
        kernel_size: int,
        padding: int,
        num_groups: int,
    ):
        # num_blocks is n in wide resnet paper
        # how many layers each block has
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        # print(f"Strides: {strides}")

        for stride in strides:
            layers.append(
                block(
                    self.field_type,
                    out_channels,
                    stride=stride,
                    frequency=frequency,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_groups=num_groups,
                )
            )
            self.field_type = layers[-1].out_type

        return SequentialModule(*layers)

    def forward(self, x):
        # Wrap input tensor in a GroupTensor
        x = GroupTensor(x, self.input_field_type)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.restrict(x)
        x = self.layer3(x)
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def iter_fix_param(self, l, eq_conv_block, normal_conv_block, n=None, stride=None):
        #print(f"\nFixing parameters for layer {l}")
        # large number
        last_ratio = 1e8
        norm_para = sum([p.numel() for p in normal_conv_block.parameters() if p.requires_grad])
        sign_switch = None
        old_sign = None
        while True:
            equi_param = sum([p.numel() for p in eq_conv_block.parameters() if p.requires_grad])
            # get the sign and ratio
            sign = np.sign(equi_param - norm_para)
            if old_sign != None:
                sign_switch = True if sign != old_sign else False
            current_ratio = equi_param / norm_para
            # if current_ratio is in 5% range, break
            # print(f"Current ratio: {current_ratio}, last ratio: {last_ratio}")

            if abs(current_ratio - 1) < 0.05 or sign_switch:
                break
            # if not, change the number of channels
            self.num_channels[l] -= sign
            old_eq_conv_block = eq_conv_block
            if l == 0:
                # change the conv1
                eq_conv_block = EquivariantConvBlock(
                    in_type=self.input_field_type,
                    out_channels=int(self.num_channels[l]),
                    frequency=self.rotation,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    num_groups=self.num_groups[l],
                )
            else:
                eq_conv_block = EquivariantWideConvBlock(
                    self.field_type,
                    out_channels=self.num_channels[l],
                    stride=stride,
                    frequency=self.rotation,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    num_groups=self.num_groups[l],
                )
            
            last_ratio = current_ratio
            old_sign = sign
            # print(f'equi_conv_block_params: {equi_param}, normal_conv_block_params: {norm_para}, ratio: {last_ratio}')
        
        if abs(last_ratio - 1) < abs(current_ratio - 1):
            eq_conv_block = old_eq_conv_block
        equi_param = sum([p.numel() for p in eq_conv_block.parameters() if p.requires_grad])
        last_ratio = equi_param / norm_para
        #print(f'fixed block ratio {last_ratio}')
        return eq_conv_block

def wide_layer(
    field_type,
    block,
    out_channels: int,
    num_blocks: int,
    stride: int,
    frequency: int,
    kernel_size: int,
    padding: int,
    num_groups: int,
):
    # num_blocks is n in wide resnet paper
    # how many layers each block has
    strides = [stride] + [1] * (int(num_blocks) - 1)
    layers = []
    # print(f"Strides: {strides}")
    for stride in strides:
        layers.append(
            block(
                field_type,
                out_channels,
                stride=stride,
                frequency=frequency,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups,
            )
        )
        field_type = layers[-1].out_type
    return SequentialModule(*layers)



FIX_PARAM_DICT = {
    3: 0.8889,
    5: 0.84,
    7: 0.8163,
    9: 0.8025,
}

def calculate_fixed_params(num_channels, kernel_size, group, gspace, rotation, restrict):
    # deepcopy to avoid changing the original list
    num_channels = num_channels.copy()
    layout = num_channels.copy()
    for l in range(len(num_channels)):
        if group == "orthogonal":
            num_channels[l] = int(num_channels[l] / (rotation + 0.9))
        else:
            num_channels[l] = int(
                        (
                            num_channels[l]
                            * np.sqrt(1.25 * gspace.fibergroup.rotation_order)
                        )
                        / gspace.fibergroup.rotation_order
                    )
        if num_channels[l] < 1:
            warnings.warn(
                f"Group order ({gspace.fibergroup.rotation_order}) is larger"
                f" than number of channels ({num_channels[l]}) defined in layout!"
            )
            num_channels[l] = 1
    

    if restrict == "halved":
        print("Group order: ", gspace.fibergroup.rotation_order)
        reduction = FIX_PARAM_DICT[kernel_size] * (gspace.fibergroup.rotation_order / 2)
        if group == "dihedral":
            reduction *= 2
        num_channels[3] = int(round(
            layout[3]                 
            / reduction, 0
            ))
        if num_channels[3] < 1:
            warnings.warn(
                f"Group order ({gspace.fibergroup.rotation_order/2}) is larger"
                f" than number of channels ({layout[2]}) defined in layout!"
            )
            num_channels[3] = 1
    elif restrict == "reflection":
        num_channels[3] = int(num_channels[2] * np.sqrt(3) / 2)
    elif restrict == "invariant":
        num_channels[3] = layout[3]
    
    return num_channels


@hydra.main(config_path="../experiment/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(f"Kernel layout: ", cfg.model.kernel_layout)
    inp = torch.rand(1, 1, 32, 32)
    n_inputs = inp.shape[1]
    n_outputs = 10
    # depth, num_classes, widen_factor=1, dropRate=0.0
    net = hydra.utils.instantiate(
            cfg.model,
            input_channels=n_inputs,
            num_classes=n_outputs,
        )
    # tot_param = sum([p.numel() for p in net.conv1.parameters()  if p.requires_grad])
    tot_param = sum([p.numel() for p in net.parameters()  if p.requires_grad])
    print('Total number of parameters: {}'.format(tot_param)) # total 2.748.890 # block1 121248
    #print(net.layer1)

    inp = inp.cuda()
    net.cuda()
    print(net(inp).size())

    #y = net(torch.randn(1,3,32,32))
    #print(y.size())


if __name__ == "__main__":
    main()