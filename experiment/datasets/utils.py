
def get_normalize_weights(df):
    total_images = df.shape[0]
    weights_loss = []

    for c in df['label'].unique():
        samples = df['label'].value_counts()[c]
        weights_loss.append(1 / (samples / total_images))

    normalized_weights = [x / sum(weights_loss) for x in weights_loss]
    return normalized_weights