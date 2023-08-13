import torch
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

import logging
from ax.utils.common.logger import ROOT_STREAM_HANDLER
ROOT_STREAM_HANDLER.setLevel(logging.ERROR)


class Search_Space:
    def __init__(self, num_middle_blocks=3):
        self.num_middle_blocks = num_middle_blocks
        self.choice_2_range_params = {"group": [1, 2, 4, 8, 16]}
        self.parameters = []

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
        stride_constraint = f"{strides_constraint} >= {6}"
        parameter_constraints.append(stride_constraint)
            
        # out_channels constraint
        out_channels_constraint += f" <= {6}"
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
            "bounds": [-1, 0],
            "value_type": "int",
        }

    def get_group(self, block_id):
        return {
            "name": f"{block_id}_group",
            "type": "range",
            "bounds": [0, len(self.choice_2_range_params["group"]) - 1],
            "value_type": "int",
        }

    def get_num_layers(self, block_id):
        return {
            "name": f"{block_id}_num_layers",
            "type": "range",
            "bounds": [1,2],
            "value_type": "int",
        }

    def get_conv_op(self, block_id):
        return {
            "name": f"{block_id}_conv_op",
            "type": "choice",
            "values": ["conv", "dconv", "mbconv"],
            "value_type": "str",
            "is_ordered": True,
        }

    def get_kernel_size(self, block_id):
        return {
            "name": f"{block_id}_kernel_size",
            "type": "choice",
            "values": [3,5],
            "value_type": "int",
            "is_ordered": True,
        }

    def get_se_ratio(self, block_id):
        return {
            "name": f"{block_id}_se_ratio",
            "type": "choice",
            "values": [0.25, 0.5, 0.75],
            "value_type": "float",
            "is_ordered": True,
        }

    def get_skip_op(self, block_id):
        return {
            "name": f"{block_id}_skip_op",
            "type": "choice",
            "values": ["no", "identity", "conv"],
            "value_type": "str",
            "is_ordered": True,
        }

    def get_out_channels(self, block_id):
        return {
            "name": f"{block_id}_out_channels",
            "type": "range",
            "bounds": [1.,4.],
            "value_type": "float",
            "is_ordered": True,
        }
        
    def get_stride(self, block_id):
        return {
            "name": f"{block_id}_stride",
            "type": "range",
            "bounds": [1, 2],
            "value_type": "int",
            "is_ordered": True,
        }
        

    def get_expand_ratio(self):
        return {
            "name": f"-1_expand_ratio",
            "type": "choice",
            "values": [2,4,6],
            "value_type": "int",
            "is_ordered": True,
        }
    
    def get_dropout_rate(self):
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


def generate_random_data():
    # generate random data
    data = {
            "valid_acc": np.random.uniform(0.7, 0.9),
            "gflops": np.random.uniform(100, 200),
            "model_building_time": np.random.uniform(10, 100),
        }
    return data


def main():
        device = torch.device('cuda' if torch.cuda.is_available() \
                                   else "cpu")
        generation_strategy=GenerationStrategy(
            name="SAASBO",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=1
                ),
                GenerationStep(
                    model=Models.FULLYBAYESIANMOO,
                    num_trials=25,
                    model_kwargs={
                        "torch_device": device,
                        "num_samples": 256,
                        "warmup_steps": 512,
                    },
                    max_parallelism=1,
                )
            ],
        )

        ax_client = AxClient(
                generation_strategy=generation_strategy,
            )

        eq_search_space = Search_Space()
        parameter = eq_search_space.get_parameters()
        parameter_constraints = eq_search_space.get_parameter_constraints()

        ax_client.create_experiment(
            parameters=parameter,
            support_intermediate_data=True,
            objectives={
                # `threshold` arguments are optional
                "valid_acc": ObjectiveProperties(
                    minimize=False, 
                    threshold=0.8
                ), 
                "gflops": ObjectiveProperties(
                    minimize=True, 
                    threshold=150
                )
            },
            parameter_constraints=parameter_constraints,
            tracking_metric_names=["model_building_time"],
        )

        for i in range(30):
            print(f"Running trial {i}")
            trial = ax_client.get_next_trial()
            _, trial_index = trial

            data = generate_random_data()

            ax_client.complete_trial(
                    trial_index=trial_index, 
                    raw_data=data,
                )
            
        
if __name__ == "__main__":
    main()