import math
import sys
import os
from typing import Tuple
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table, parameter_count
from matplotlib import pyplot as plt

sys.path.append(f"{os.getcwd()}")
import torch
from nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    GroupTensor,
    R2Conv,
    GroupNorm,
    Mish,
    GroupPooling,
    EquivariantModule,
)
from networks.util import get_group_id, get_gspace_from_id

class TestNet(EquivariantModule):
    def __init__(self, input_channels: int = 5, gspace = None):
        super().__init__()
        
        self.input_field = FieldType(
                    gspace, [gspace.regular_repr] * input_channels
                )
        self.conv = R2Conv(
                self.input_field,
                self.input_field,
                kernel_size=3,
                padding=1,
                groups=1,
                stride=1,
                dilation=1,
                bias=False,
                frequencies_cutoff=lambda r: 3 * r,
            )
        

    def forward(self, input):
        input = GroupTensor(input, self.input_field)
        out = self.conv(input)
        return out.tensor
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.original_in_type.size
        return input_shape

def main(rotations: list = [1, 2,4,8,16], reflections = [-1,0], input_channels: int = 64):
    data = {}
    for mode in ["ad", "no"]:
        data[mode] = {}
        for ref in reflections:
            data[mode][ref] = {}
            for rot in rotations:
                group_id = get_group_id(ref, rot)
                gspace = get_gspace_from_id(group_id)
                print(f"\nNEXT: Group {gspace.fibergroup}, mode: {mode}")

                N = gspace.fibergroup.order()
                in_channels = (input_channels / N)
                if mode == "ad":
                    in_channels = in_channels * math.sqrt(N)
                print(f"Input channels: {in_channels}")
                in_channels = int(in_channels)


                model = TestNet(input_channels=in_channels, gspace=gspace).cuda()

                param_count = parameter_count(model)["conv"]
                print(f"Param count: {param_count}")

                multiplier = rot
                multiplier *= 2 if ref == 0 else 1
                input_tensor = torch.randn(1, in_channels * multiplier, \
                32, 32).cuda()
                flops = FlopCountAnalysis(model, (input_tensor,))
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                flops = flops.total()
                print(f"Flops: {flops}")

                data[mode][ref][rot] = (param_count, flops)

    visualize_data(data)


def visualize_data(data):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    ad_data = data['ad']
    no_data = data['no']

    # Ref -1 - Flops
    axs[0, 0].plot(ad_data[-1].keys(), [item[1] for item in ad_data[-1].values()], 'r-o', label='AD')
    axs[0, 0].plot(no_data[-1].keys(), [item[1] for item in no_data[-1].values()], 'g-o', label='NO')
    axs[0, 0].set_title('No Reflection - Flops')
    axs[0, 0].set_xlabel('Rotations')
    axs[0, 0].set_ylabel('Flops')
    axs[0, 0].legend()

    # Ref 0 - Flops
    axs[0, 1].plot(ad_data[0].keys(), [item[1] for item in ad_data[0].values()], 'b-o', label='AD')
    axs[0, 1].plot(no_data[0].keys(), [item[1] for item in no_data[0].values()], 'm-o', label='NO')
    axs[0, 1].set_title('With Reflection - Flops')
    axs[0, 1].set_xlabel('Rotations')
    axs[0, 1].set_ylabel('Flops')
    axs[0, 1].legend()

    # Ref -1 - Param Count
    axs[1, 0].plot(ad_data[-1].keys(), [item[0] for item in ad_data[-1].values()], 'r-o', label='AD')
    axs[1, 0].plot(no_data[-1].keys(), [item[0] for item in no_data[-1].values()], 'g-o', label='NO')
    axs[1, 0].set_title('No Reflection - Param Count')
    axs[1, 0].set_xlabel('Rotations')
    axs[1, 0].set_ylabel('Param Count')
    axs[1, 0].legend()

    # Ref 0 - Param Count
    axs[1, 1].plot(ad_data[0].keys(), [item[0] for item in ad_data[0].values()], 'b-o', label='AD')
    axs[1, 1].plot(no_data[0].keys(), [item[0] for item in no_data[0].values()], 'm-o', label='NO')
    axs[1, 1].set_title('With Reflection - Param Count')
    axs[1, 1].set_xlabel('Rotations')
    axs[1, 1].set_ylabel('Param Count')
    axs[1, 1].legend()

    plt.savefig('scaling/figures/param_vs_flops.png')



if __name__ == "__main__":
    main()