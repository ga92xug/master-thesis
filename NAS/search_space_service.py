from ax import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
    ParameterConstraint,
)
from ax import ParameterType, RangeParameter, SearchSpace
from ax.core import ParameterConstraint, OrderConstraint
import numpy as np
from omegaconf import OmegaConf
import warnings

from requests import get
from torch import mul
warnings.filterwarnings("ignore", category=UserWarning)


class Eq_Search_Space:
    def __init__(self, search_space_cfg):
        self.search_space_cfg = search_space_cfg
        self.num_middle_blocks = search_space_cfg.num_middle_blocks
        choice_2_range_params = OmegaConf.to_container(search_space_cfg.choice_2_range_params, resolve=True)
        self.choice_2_range_params = choice_2_range_params
        self.parameters = []
        # global params
        self.parameters.extend([
            self.get_expand_ratio(),
            self.get_dropout_rate(),
        ])

        # block search space
        # block_id = 0 is the stem block
        # block_id = blocks+1 is the head block
        for block_id in range(0, self.num_middle_blocks+2):
            self.parameters.extend(self.per_block_search_space(block_id))

        self.parameter_constraints = self.get_constraints()

    def get_constraints(self):       
        parameter_constraints = []
        strides_constraint = ""
        out_channels_constraint = ""
        total_number_blocks = self.num_middle_blocks+2
        out_channels_prefactor = np.linspace(1, 0.1, total_number_blocks)

        for block_id in range(0, total_number_blocks):
            if block_id == 0:
                strides_constraint += f"{block_id}_stride"
                out_channels_constraint += f"{out_channels_prefactor[block_id]}*{block_id}_out_channels"
                continue

            # the group must never increase
            group_decrease_constraint = f"{block_id-1}_group >= {block_id}_group"
            parameter_constraints.append(group_decrease_constraint)
            
            # the reflection must never increase
            reflection_decrease_constraint = f"{block_id-1}_reflection >= {block_id}_reflection"
            parameter_constraints.append(reflection_decrease_constraint)

            strides_constraint += f" + {block_id}_stride"
            out_channels_constraint += f" + {out_channels_prefactor[block_id]}*{block_id}_out_channels"

        # stride constraint
        try:
            stride_constraint = f"{strides_constraint} >= {self.search_space_cfg.constraints.min_stride}"
            parameter_constraints.append(stride_constraint)
        except:
            try:
                tmp = self.search_space_cfg.strides
            except:
                raise Exception("We either need to define stride_constraint or strides in the config")
            
        # out_channels constraint
        out_channels_constraint += f" <= {self.search_space_cfg.constraints.max_out_channels}"
        parameter_constraints.append(out_channels_constraint)

        return parameter_constraints

    def per_block_search_space(self, block_id):
        block_search_space = []

        if block_id == 0:
            block_search_space.extend([
                self.get_reflection(block_id),
                self.get_group(block_id),
                self.get_out_channels(block_id),
                self.get_kernel_size(block_id),
                self.get_stride(block_id),
            ])
                                      
            return block_search_space
        
        elif block_id == self.num_middle_blocks+1:
            block_search_space.extend([
                self.get_reflection(block_id),
                self.get_group(block_id),
                self.get_out_channels(block_id),
                self.get_kernel_size(block_id),
                self.get_stride(block_id),
            ])
            return block_search_space
        
        else:
            block_search_space.extend([
                self.get_reflection(block_id),
                self.get_group(block_id),
                self.get_num_layers(block_id),
                self.get_conv_op(block_id),
                self.get_kernel_size(block_id),
                self.get_se_ratio(block_id),
                self.get_out_channels(block_id),
                self.get_skip_op(block_id),
                self.get_stride(block_id),
            ])
            return block_search_space


    def get_reflection(self, block_id):
        return {
            "name": f"{block_id}_reflection",
            "type": "range",
            "bounds": list(self.search_space_cfg.reflection),
            "value_type": "int",
        }

    def get_group(self, block_id):
        # the first block must have a group of at least 1
        if block_id == 0 and self.choice_2_range_params["group"][0] == 0:
            lower_bound = 1
        else:
            lower_bound = 0 
        return {
            "name": f"{block_id}_group",
            "type": "range",
            "bounds": [lower_bound, len(self.choice_2_range_params["group"]) - 1],
            "value_type": "int",
        }

    def get_num_layers(self, block_id):
        return {
            "name": f"{block_id}_num_layers",
            "type": "range",
            "bounds": list(self.search_space_cfg.num_layers),
            "value_type": "int",
        }

    def get_conv_op(self, block_id):
        return {
            "name": f"{block_id}_conv_op",
            "type": "choice",
            "values": list(self.search_space_cfg.conv_op),
            "value_type": "str",
            "is_ordered": True,
        }

    def get_kernel_size(self, block_id):
        if block_id == self.num_middle_blocks+1:
            kernel_sizes = [0]
            kernel_sizes.extend(list(self.search_space_cfg.kernel_size))
        else:
            kernel_sizes = list(self.search_space_cfg.kernel_size)

        return {
            "name": f"{block_id}_kernel_size",
            "type": "choice",
            "values": kernel_sizes,
            "value_type": "int",
            "is_ordered": True,
        }

    def get_se_ratio(self, block_id):
        return {
            "name": f"{block_id}_se_ratio",
            "type": "choice",
            "values": list(self.search_space_cfg.se_ratio),
            "value_type": "float",
            "is_ordered": True,
        }

    def get_skip_op(self, block_id):
        return {
            "name": f"{block_id}_skip_op",
            "type": "choice",
            "values": list(self.search_space_cfg.skip_op),
            "value_type": "str",
            "is_ordered": True,
        }

    def get_out_channels(self, block_id):
        return {
            "name": f"{block_id}_out_channels",
            "type": "range",
            "bounds": list(self.search_space_cfg.out_channels),
            "value_type": "float",
            "is_ordered": True,
        }
        

    def get_stride(self, block_id):
        try:
            # if we have declared strides in config, use them
            strides = list(self.search_space_cfg.strides)
            stride = strides[block_id]
            return {
                "name": f"{block_id}_stride",
                "type": "fixed",
                "value": stride,
                "value_type": "int",
            }
        except:
            return {
                "name": f"{block_id}_stride",
                "type": "range",
                "bounds": list(self.search_space_cfg.stride),
                "value_type": "int",
                "is_ordered": True,
            }
        

    def get_expand_ratio(self):
        try:
            return {
                "name": f"-1_expand_ratio",
                "type": "choice",
                "values": list(self.search_space_cfg.expand_ratio),
                "value_type": "int",
                "is_ordered": True,
            }
        except:
            return {
                "name": f"-1_expand_ratio",
                "type": "fixed",
                "value": 2,
                "value_type": "int",
            }
    
    def get_dropout_rate(self):
        try:
            return {
                "name": f"-1_dropout_rate",
                "type": "choice",
                "values": list(self.search_space_cfg.dropout_rate),
                "value_type": "float",
                "is_ordered": True,
            }
        except:
            return {
                "name": f"-1_dropout_rate",
                "type": "fixed",
                "value": 0.0,
                "value_type": "float",
            }
        
    def get_search_space(self):
        return self.search_space
    
    def get_parameters(self):
        return self.parameters
    
    def get_parameter_constraints(self):
        return self.parameter_constraints


def dict_to_list(d):
    result = []
    for value in d.values():
        #print(value)
        if isinstance(value, dict):
            result.extend(dict_to_list(value))
        else:
            result.append(value)
    return result

