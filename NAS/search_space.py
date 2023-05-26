
from ax import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
    ParameterConstraint,
)
from ax import ParameterType, RangeParameter, SearchSpace
from ax.core.constraint import ParameterConstraint, OrderConstraint

######################################################################
# Setting up the ``SearchSpace``
# ------------------------------
#
# First, we define our search space. Ax supports both range parameters
# of type integer and float as well as choice parameters which can have
# non-numerical types such as strings.
# We will tune the hidden sizes, learning rate, dropout, and number of
# epochs as range parameters and tune the batch size as an ordered choice
# parameter to enforce it to be a power of 2.
#
class Eq_Search_Space:
    def __init__(self, blocks=3):
        self.blocks = blocks
        self.parameters = []


        # initial reflection
        initial_reflection = ChoiceParameter(
            name=f"0_reflection",
            values=[-1, 0],
            parameter_type=ParameterType.INT,
            is_ordered=True,
            sort_values=True,
        )
        # initial group
        initial_group = ChoiceParameter(
            name=f"0_group",
            values=[1, 2, 4, 8, 12, 16],
            parameter_type=ParameterType.INT,
            is_ordered=True,
            sort_values=True,
        )
        
        # initial out_channels
        initial_out_channels = ChoiceParameter(
            name=f"0_out_channels",
            values=[1, 2, 4, 8, 16, 32],
            parameter_type=ParameterType.INT,
            is_ordered=True,
            sort_values=True,
        )

        # initial kernel_size
        initial_kernel_size = ChoiceParameter(
            name=f"0_kernel_size",
            values=[3, 5],
            parameter_type=ParameterType.INT,
            is_ordered=True,
            sort_values=True,
        )

        self.parameters.extend([initial_reflection, initial_group, 
                                initial_out_channels, initial_kernel_size])

        # block search space
        for block_id in range(1, self.blocks + 1):
            self.parameters.extend(self.per_block_search_space(block_id))

        constraints = self.get_constraints()

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
        group_decrease = [
            OrderConstraint(
                    upper_parameter = f"{i-1}_group",
                    lower_parameter = f"{i}_group",
            ) for i in range(1, self.blocks)
        ]

        reflection_decrease = [
            OrderConstraint(
                    upper_parameter = f"{i-1}_reflection",
                    lower_parameter = f"{i}_reflection",
            ) for i in range(1, self.blocks)
        ]

        return group_decrease + reflection_decrease 

    def per_block_search_space(self, block_id):
        return [
            # Reflection
            ChoiceParameter(
                name=f"{block_id}_reflection",
                values=[-1, 0],
                parameter_type=ParameterType.INT,
                is_ordered=True,
                sort_values=True,
            ),

            # Group
            ChoiceParameter(
                name=f"{block_id}_group",
                values=[1, 2, 4, 8, 12, 16],
                parameter_type=ParameterType.INT,
                is_ordered=True,
                sort_values=True,
            ),

            # num_layers
            RangeParameter(
                name=f"{block_id}_num_layers",
                lower=1,
                upper=3,
                parameter_type=ParameterType.INT,
                log_scale=False,
            ),
            # ConvOp
            ChoiceParameter(
                name=f"{block_id}_conv_op",
                values=["conv", "dconv", "mobilenet"],
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            ),
            # KernelSize
            ChoiceParameter(
                name=f"{block_id}_kernel_size",
                values=[3, 5],
                parameter_type=ParameterType.INT,
                is_ordered=True,
                sort_values=True,
            ),
            # SERatio
            ChoiceParameter(
                name=f"{block_id}_se_ratio",
                values=[0, 0.25],
                parameter_type=ParameterType.FLOAT,
                is_ordered=True,
                sort_values=True,
            ),
            # skip connection
            ChoiceParameter(
                name=f"{block_id}_skip_op",
                values=["identity", "no"], # "pool"
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            ),
            # out_channels
            ChoiceParameter(
                name=f"{block_id}_out_channels",
                values=[1.,1.25,1.5,1.75,2.],
                parameter_type=ParameterType.FLOAT,
                is_ordered=True,
                sort_values=True,
            ),
        ]
