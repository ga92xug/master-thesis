import numpy as np
import pandas as pd

def get_normalize_weights(labels):
    df = pd.DataFrame({"label": labels})
    weights = df['label'].value_counts() / df['label'].value_counts().sum()
    # sort based on index
    weights = weights.sort_index()
    assert np.isclose(weights.sum(), 1.0), "weights should sum to 1.0"
    weights = weights.values
    print("weights: ", weights)
    return weights