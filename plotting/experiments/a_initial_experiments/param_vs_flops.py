import math
import sys
import os
from typing import Tuple
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table, parameter_count
from matplotlib import pyplot as plt
from matplotlib.artist import setp
import torch
from joblib import Memory
from matplotlib.ticker import MaxNLocator


sys.path.append(f"{os.getcwd()}")
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
from networks.util import (
    get_group_id, 
    get_gspace_from_id,
    adjusted_out_channels,
)

from plotting.experiments.plotting_utils import *

class CNN_TestNet(torch.nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, input):
        out = self.conv(input)
        return out

class EQ_TestNet(EquivariantModule):
    def __init__(self, input_channels: int, gspace, fixed_params: bool = True):
        super().__init__()

        self.adjusted_channels = adjusted_out_channels(
            out_channel=input_channels,
            N=gspace.fibergroup.order(),
            fixed_params=fixed_params,
        )
        
        self.input_field = FieldType(
                    gspace, [gspace.regular_repr] * self.adjusted_channels
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

cache_dir = '/home/frischs/.cache/param_vs_flops/'
memory = Memory(location=cache_dir, verbose=1)
@memory.cache
def get_data_eq_conv(input_channels: int, rotations: list, reflections: list):
    data = {}
    for mode in ["ad", "no"]:
        data[mode] = {}
        for ref in reflections:
            data[mode][ref] = {}
            for rot in rotations:
                group_id = get_group_id(ref, rot)
                gspace = get_gspace_from_id(group_id)
                print(f"\nNEXT: Group {gspace.fibergroup}, mode: {mode}")

                fix_params = True if mode == "ad" else False
                model = EQ_TestNet(input_channels=input_channels, gspace=gspace, fixed_params=fix_params).cuda()

                param_count = parameter_count(model)["conv"]
                print(f"Param count: {param_count}")

                multiplier = rot
                multiplier *= 2 if ref == 0 else 1
                input_tensor = torch.randn(1, model.adjusted_channels * multiplier, \
                32, 32).cuda()
                flops = FlopCountAnalysis(model, (input_tensor,))
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                flops = flops.total()
                print(f"Flops: {flops}")

                data[mode][ref][rot] = (param_count, flops)

    return data

@memory.cache
def get_data_cnn_conv(input_channels: int):
    model = CNN_TestNet(input_channels=input_channels).cuda()
    param_count = parameter_count(model)["conv"]
    input_tensor = torch.randn(1, input_channels, 32, 32).cuda()
    flops = FlopCountAnalysis(model, (input_tensor,))
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops = flops.total()
    data = (param_count, flops)
    return data


def reduce_data(data_eq_conv: dict, data_cnn_conv: dict, param_count_in: int, flops_in: int):
    param_divisor = 10 ** param_count_in
    flops_divisor = 10 ** flops_in
    data_out = {}
    #print(data_eq_conv)
    for mode, mode_data in data_eq_conv.items():
        data_out[mode] = {}
        for ref, ref_data in mode_data.items():
            data_out[mode][ref] = {}
            for rot, rot_data in ref_data.items():

                param_count, flops = rot_data

                param_count /= param_divisor
                flops /= flops_divisor

                data_out[mode][ref][rot] = (param_count, flops)

    param_count, flops = data_cnn_conv
    param_count /= param_divisor
    flops /= flops_divisor
    data_cnn_conv = (param_count, flops)
    return data_out, data_cnn_conv


def visualize_data(
        data_eq_conv, 
        data_cnn_conv, 
        rotations, 
        figsize=(10, 10),
        param_count_in: int = 3,
        flops_in: int = 9,
    ):
    figsize = get_fig_size(figsize)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)

    axs[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=1, integer=True))

    data_eq_conv, data_cnn_conv = reduce_data(data_eq_conv, data_cnn_conv, param_count_in, flops_in)    

    ad_data = data_eq_conv['ad']
    no_data = data_eq_conv['no']

    # Ref -1 - Flops
    axs[0, 0].plot(ad_data[-1].keys(), [item[1] for item in ad_data[-1].values()], 'r-o', label='Equivariant Convolution Parameter Adjusted')
    axs[0, 0].plot(no_data[-1].keys(), [item[1] for item in no_data[-1].values()], 'g-o', label='Equivariant Convolution FLOPs Adjusted')
    axs[0, 0].set_title('Cyclic Group')
    axs[0, 0].set_ylabel(f'FLOPs [$10^{flops_in}$]')
    setp(axs[0, 0].get_xticklabels(), visible=False)
    axs[0, 0].grid(True)

    # Ref 0 - Flops
    axs[0, 1].plot(ad_data[0].keys(), [item[1] for item in ad_data[0].values()], 'r-o', label='Equivariant Convolution Parameter Adjusted')
    axs[0, 1].plot(no_data[0].keys(), [item[1] for item in no_data[0].values()], 'g-o', label='Equivariant Convolution FLOPs Adjusted')
    axs[0, 1].set_title('Dihedral Group')
    axs[0, 1].grid(True)
    setp(axs[0, 1].get_xticklabels(), visible=False)
    setp(axs[0, 1].get_yticklabels(), visible=False)
    

    # Ref -1 - Param Count
    axs[1, 0].plot(ad_data[-1].keys(), [item[0] for item in ad_data[-1].values()], 'r-o', label='AD')
    axs[1, 0].plot(no_data[-1].keys(), [item[0] for item in no_data[-1].values()], 'g-o', label='NO')
    axs[1, 0].set_xlabel('Rotations')
    axs[1, 0].set_ylabel(f'Parameters [$10^{param_count_in}$]')
    axs[1, 0].set_xticks(rotations)
    axs[1, 0].grid(True)

    # Ref 0 - Param Count
    axs[1, 1].plot(ad_data[0].keys(), [item[0] for item in ad_data[0].values()], 'r-o', label='AD')
    axs[1, 1].plot(no_data[0].keys(), [item[0] for item in no_data[0].values()], 'g-o', label='NO')
    axs[1, 1].set_xlabel('Rotations')
    axs[1, 1].set_xticks(rotations)
    axs[1, 1].grid(True)
    setp(axs[1, 1].get_yticklabels(), visible=False)
    

    if data_cnn_conv is not None:
        # add line for cnn
        param_count, flops = data_cnn_conv
        axs[0, 0].axhline(y=flops, color='b', linestyle=':', label='CNN')
        axs[0, 1].axhline(y=flops, color='b', linestyle=':', label='CNN')
        axs[1, 0].axhline(y=param_count, color='b', linestyle=':', label='CNN')
        axs[1, 1].axhline(y=param_count, color='b', linestyle=':', label='CNN')

    axs[0, 0].sharey(axs[0, 1])
    axs[1, 0].sharey(axs[1, 1])
    axs[0, 0].sharex(axs[1, 0])
    axs[0, 1].sharex(axs[1, 1])

    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.11, 0.95))
    #axs[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    plt.tight_layout()
    return fig


def main(input_channels: int = 64, rotations: list = [1,2,4,6,8,10,12,14,16], reflections: list = [-1,0]):
    cfg, save_folder_name = plot_init("a_initial_experiments/param_vs_flops/")

    data_eq_conv = get_data_eq_conv(input_channels, rotations, reflections)
    data_cnn_conv = get_data_cnn_conv(input_channels)

    fig = visualize_data(data_eq_conv, data_cnn_conv, rotations)

    save_plot(
        figure=fig,
        name="param_vs_flops",
        folder_name=save_folder_name,
    )


if __name__ == "__main__":
    
    main()
    #    input_channels=16, 
    #    rotations=[1,2,4,6,8], 
    #    reflections=[-1,0],
    #)