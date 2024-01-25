
from typing import List


def write_predictions_2_txt(predictions: List[str]):
    # Process and save predictions
    with open('submission.txt', 'w') as file:
        for prediction in predictions:
            # Format the prediction as needed (e.g., image_id, predicted_label)
            file.write(formatted_prediction + '\n')