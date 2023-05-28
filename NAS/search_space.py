
from ax import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
    ParameterConstraint,
)
from ax import ParameterType, RangeParameter, SearchSpace
from ax.core import ParameterConstraint, OrderConstraint


class Eq_Search_Space:
    def __init__(self, choice_2_range_params, blocks=3):
        self.blocks = blocks
        self.choice_2_range_params = choice_2_range_params
        self.parameters = {}

        # block search space
        # block_id = 0 is the stem block
        # block_id = blocks+1 is the head block
        for block_id in range(0, self.blocks+1):
            self.parameters[block_id] = self.per_block_search_space(block_id)


        constraints = self.get_constraints()

        # convert dict to list
        self.parameters = dict_to_list(self.parameters)

        self.search_space = SearchSpace(parameters=self.parameters,
                                        parameter_constraints=constraints)

    def get_constraints(self):
        # currently output channels increase by [1.0, 2.0] at every block
        # filter_size_increase = \
        # [
        #     OrderConstraint(
        #             lower_parameter = f"{i-1}_filter_size",
        #             upper_parameter = f"{i}_filter_size",
        #     ) for i in range(1, self.blocks)
        # ]
        group_decrease = []
        reflection_decrease = []
        for block_id, block_param_dict in self.parameters.items():
            if block_id == 0:
                # there is no constraint for the first block
                continue
            group_decrease.append(OrderConstraint(
                    upper_parameter = self.parameters[block_id-1]["group"],
                    lower_parameter = self.parameters[block_id]["group"],
            ))
            reflection_decrease.append(OrderConstraint(
                    upper_parameter = self.parameters[block_id-1]["reflection"],
                    lower_parameter = self.parameters[block_id]["reflection"],
            ))

        return group_decrease + reflection_decrease 

    def per_block_search_space(self, block_id):
        block_search_space = {}

        if block_id == 0:
            block_search_space["reflection"] = self.get_reflection(block_id)
            block_search_space["group"] = self.get_group(block_id)
            block_search_space["out_channels"] = self.get_out_channels(block_id)
            block_search_space["kernel_size"] = self.get_kernel_size(block_id)
            return block_search_space
        
        elif block_id == self.blocks+1:
            block_search_space["reflection"] = self.get_reflection(block_id)
            block_search_space["group"] = self.get_group(block_id)
            block_search_space["out_channels"] = self.get_out_channels(block_id)
            block_search_space["kernel_size"] = self.get_kernel_size(block_id)
            return block_search_space
        
        else:
            block_search_space["reflection"] = self.get_reflection(block_id)
            block_search_space["group"] = self.get_group(block_id)
            block_search_space["num_layers"] = self.get_num_layers(block_id)
            block_search_space["conv_op"] = self.get_conv_op(block_id)
            block_search_space["kernel_size"] = self.get_kernel_size(block_id)
            block_search_space["se_ratio"] = self.get_se_ratio(block_id)
            block_search_space["out_channels"] = self.get_out_channels(block_id)
            block_search_space["skip_op"] = self.get_skip_op(block_id)
            return block_search_space


    def get_reflection(self, block_id):
        return RangeParameter(
                name=f"{block_id}_reflection",
                lower=-1,
                upper=0,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_group(self, block_id):
        return RangeParameter(
                name=f"{block_id}_group",
                lower=0,
                upper=len(self.choice_2_range_params["group"])-1,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_num_layers(self, block_id):
        return RangeParameter(
                name=f"{block_id}_num_layers",
                lower=1,
                upper=3,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_conv_op(self, block_id):
        return ChoiceParameter(
                name=f"{block_id}_conv_op",
                values=["conv", "dconv", "mbconv"],
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            )
    
    def get_kernel_size(self, block_id):
        return RangeParameter(
                name=f"{block_id}_kernel_size",
                lower=0,
                upper=len(self.choice_2_range_params["kernel_size"])-1,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_se_ratio(self, block_id):
        return RangeParameter(
                name=f"{block_id}_se_ratio",
                lower=0,
                upper=len(self.choice_2_range_params["se_ratio"])-1,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_skip_op(self, block_id):
        return ChoiceParameter(
                name=f"{block_id}_skip_op",
                values=["identity", "no"], # "pool"
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            )
    
    def get_out_channels(self, block_id):
        return RangeParameter(
                name=f"{block_id}_out_channels",
                lower=0,
                upper=len(self.choice_2_range_params["out_channels"])-1,
                parameter_type=ParameterType.INT,
                log_scale=False,
            )
    
    def get_search_space(self):
        return self.search_space


def dict_to_list(d):
    result = []
    for value in d.values():
        #print(value)
        if isinstance(value, dict):
            result.extend(dict_to_list(value))
        else:
            result.append(value)
    return result