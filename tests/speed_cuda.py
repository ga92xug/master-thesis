import sys
import os
import torch
import torch.utils.benchmark as benchmark
sys.path.append(f"{os.getcwd()}")
from equivariant.nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    GroupTensor,
    R2Conv,
    GroupNorm,
    Mish,
    GroupPooling,
)

# Define the operations you want to benchmark
def get_operations(operation_str: str, device: str):
    print(f"{device.upper()}: {operation_str}")
    rot=2
    in_channels=3
    gspace = flipRot2dOnR2(rot)
    input_field_type = FieldType(gspace, [gspace.regular_repr] * in_channels)

    rand_input = torch.randn(128, input_field_type.size, 128, 128)
    if device == 'gpu':
        rand_input = rand_input.cuda()
    input = GroupTensor(rand_input , input_field_type)

    if operation_str == 'R2Conv':
        operation = R2Conv(
                input_field_type, input_field_type, kernel_size=3, padding=1, groups=1, 
                stride=1, dilation=1, bias=True, frequencies_cutoff=lambda r: 3 * r
            )
    elif operation_str == 'GroupPooling':
        operation = GroupPooling(input_field_type)
    elif operation_str == 'GroupNorm':
        operation = GroupNorm(input_field_type, num_groups=3)
    elif operation_str == 'Mish':
        operation = Mish(input_field_type)
    else:
        raise ValueError(f"Operation {operation_str} not found")

    if device == 'gpu':
        operation = operation.cuda()

    # forward pass
    operation(input)

    return operation

# Benchmarking function
def benchmark_operations():
    results = []


    for name in ['R2Conv', 'GroupPooling', 'GroupNorm', 'Mish']:

        # GPU
        gpu_forward_timer = benchmark.Timer(
            stmt='get_operations(operation_str=name, device=device)',
            setup='from __main__ import get_operations',
            globals={"name": name, "device": "gpu"},
            label=name,
            description="gpu",
        )

        # CPU
        cpu_forward_timer = benchmark.Timer(
            stmt='get_operations()',
            setup='from __main__ import get_operations',
            globals={"name": name, "device": "cpu"},
            label=name,
            description="cpu",
        )

        results.append(gpu_forward_timer.timeit(5))
        #results.append(cpu_forward_timer.timeit(10))
        
    return results

# Main execution
if __name__ == "__main__":
    results = benchmark_operations()
    compare = benchmark.Compare(results)
    compare.print()