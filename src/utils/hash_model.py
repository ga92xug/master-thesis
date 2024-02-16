import torch
import lightning as L
import hashlib
import os
os.environ["PYTHONHASHSEED"] = "0"

def model_hash(model):
    # Create a SHA256 hash object
    hash_params = hashlib.sha256(usedforsecurity=False)
    hash_buffer = hashlib.sha256(usedforsecurity=False)


    # Iterate through model parameters and buffers
    for param in model.parameters():
        # Use the parameter's data converted to bytes
        hash_params.update(param.data.cpu().numpy().tobytes())

    for name, buffer in model.named_buffers():
        if "running" in name or "num_batches_tracked" in name or "indices" in name:
            continue
        print_buffer = hashlib.sha256(usedforsecurity=False)
        print_buffer.update(buffer.detach().cpu().numpy().tobytes())
        print(name, print_buffer.hexdigest()[:5])
        # Use the buffer's data converted to bytes
        hash_buffer.update(buffer.detach().cpu().numpy().tobytes())

    # Return the hexadecimal digest of the hash
    return hash_params.hexdigest(), hash_buffer.hexdigest()

def hash_state_dict(state_dict):
    # Create a SHA256 hash object
    hash_sha256 = hashlib.sha256(usedforsecurity=False)

    # Iterate through model parameters and buffers
    for key in state_dict.keys():
        # Use the parameter's data converted to bytes
        hash_sha256.update(state_dict[key].cpu().numpy().tobytes())

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()

def read_model_hash(file_path):
    state_dict = torch.load(file_path)

    net = state_dict["state_dict"]

    for key in net.keys():
        print(key)

    print(hash_state_dict(net))


if __name__ == "__main__":
    L.seed_everything(42)
    read_model_hash("pre_trained_models/last.ckpt")
    # 219ab0ea229cc01ac38f3f0226c2cf03d0110867100cc9e3eb25abceb01bca01