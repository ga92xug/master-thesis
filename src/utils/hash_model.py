import torch
import hashlib

def model_hash(model):
    # Create a SHA256 hash object
    hash_sha256 = hashlib.sha256(usedforsecurity=False)

    # Iterate through model parameters and buffers
    for param in model.parameters():
        # Use the parameter's data converted to bytes
        hash_sha256.update(param.data.cpu().numpy().tobytes())

    for buffer in model.buffers():
        # Use the buffer's data converted to bytes
        hash_sha256.update(buffer.detach().cpu().numpy().tobytes())

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()

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
    read_model_hash("pre_trained_models/last.ckpt")