
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

from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
    OrderConstraint,
)

class Eq_Search_Space:
    def __init__(self):
        pass
        

    def per_block_search_space(self, block_id):
        self.parameters = [
            # NOTE: In a real-world setting, hidden_size_1 and hidden_size_2
            # should probably be powers of 2, but in our simple example this
            # would mean that ``num_params`` can't take on that many values, which
            # in turn makes the Pareto frontier look pretty weird.
            
            # num_layers
            RangeParameter(
                name=f"num_layers_block_{block_id}",
                lower=1,
                upper=5,
                parameter_type=ParameterType.INT,
                log_scale=False,
            ),
            # ConvOp
            RangeParameter(
                name="hidden_size_1",
                lower=16,
                upper=128,
                parameter_type=ParameterType.STRING,
            ),

            # skip connection
            ChoiceParameter(
                name=f"skip_op_block_{block_id}",
                values=["identity", "pool", "no"],
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            ),

            RangeParameter(
                name="hidden_size_2",
                lower=16,
                upper=128,
                parameter_type=ParameterType.INT,
                log_scale=True,
            ),
            RangeParameter(
                name="learning_rate",
                lower=1e-4,
                upper=1e-2,
                parameter_type=ParameterType.FLOAT,
                log_scale=True,
            ),
            RangeParameter(
                name="epochs",
                lower=1,
                upper=4,
                parameter_type=ParameterType.INT,
            ),
            RangeParameter(
                name="dropout",
                lower=0.0,
                upper=0.5,
                parameter_type=ParameterType.FLOAT,
            ),

            
        ]

        self.search_space = SearchSpace(
            parameters=self.parameters,
            # NOTE: In practice, it may make sense to add a constraint
            # hidden_size_2 <= hidden_size_1
            parameter_constraints=[],
        )

        return self.search_space


    def get_search_space(self):
        return self.search_space


from ax import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
    ParameterConstraint,
)

class Eq_Search_Space:
    def __init__(self):
        self.blocks = 7
        self.parameters = []
        self.build_search_space()

    def build_search_space(self):
        for block_id in range(1, self.blocks + 1):
            self.parameters.extend(self.per_block_search_space(block_id))

        self.search_space = SearchSpace(parameters=self.parameters)

    def get_constraints(self):
        filter_size_increase = \
        [
            OrderConstraint(
                    lower_parameter = f"filter_size_block_{i}",
                    upper_parameter = f"filter_size_block_{i+1}",
            ) for i in range(1, 7)
        ]
        return filter_size_increase

    def per_block_search_space(self, block_id):
        return [
            # num_layers
            RangeParameter(
                name=f"num_layers_block_{block_id}",
                lower=1,
                upper=5,
                parameter_type=ParameterType.INT,
                log_scale=False,
            ),
            # ConvOp
            ChoiceParameter(
                name=f"conv_op_block_{block_id}",
                values=["conv", "dconv", "mobilenet"],
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            ),
            # KernelSize
            ChoiceParameter(
                name=f"kernel_size_block_{block_id}",
                values=[3, 5],
                parameter_type=ParameterType.INT,
                is_ordered=True,
                sort_values=True,
            ),
            # SERatio
            ChoiceParameter(
                name=f"se_ratio_block_{block_id}",
                values=[0, 0.25],
                parameter_type=ParameterType.FLOAT,
                is_ordered=True,
                sort_values=True,
            ),
            # skip connection
            ChoiceParameter(
                name=f"skip_op_block_{block_id}",
                values=["identity", "pool", "no"],
                parameter_type=ParameterType.STRING,
                is_ordered=False,
                sort_values=True,
            ),
            # Fi
            RangeParameter(
                name=f"filter_size_block_{block_id}",
                lower=0.75,
                upper=1.25,
                parameter_type=ParameterType.FLOAT,
                log_scale=False,
            ),
        ]
