import torch
import hashlib

def model_hash(model):
    # Create a SHA256 hash object
    hash_sha256 = hashlib.sha256()

    # Iterate through model parameters and buffers
    for param in model.parameters():
        # Use the parameter's data converted to bytes
        hash_sha256.update(param.data.cpu().numpy().tobytes())

    for buffer in model.buffers():
        # Use the buffer's data converted to bytes
        hash_sha256.update(buffer.detach().cpu().numpy().tobytes())

    # Return the hexadecimal digest of the hash
    return hash_sha256.hexdigest()
