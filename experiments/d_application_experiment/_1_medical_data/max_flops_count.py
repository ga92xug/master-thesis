
lenght_datasets = {
    "blood": 11964,
    "DeepDRiD": 1200,
}

def max_gflops_count(epochs: int, train_dataset_length: int, gflops_per_image: int):
    """
    Calculate the maximum flops count for the given epochs and dataset length.
    """
    return epochs * train_dataset_length * gflops_per_image


def main():
    """
    Calculate the maximum flops count for the given epochs and dataset length.
    """

    # we check for the average best epoch with early stopping in wandb 
    # and use this epoch for the max flops count for eq_nasnet
    epochs = 37
    dataset = "blood"
    train_dataset_length = lenght_datasets[dataset]

    gflops_per_image_efficientnet_r128_b64 = 0.137015781

    max_gflops = max_gflops_count(epochs, train_dataset_length, gflops_per_image_efficientnet_r128_b64)
    print(f"max_gflops: {max_gflops}")
    

if __name__ == "__main__":
    main()