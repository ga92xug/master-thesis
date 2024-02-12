import copy
import signal
from typing import Any, Dict, List, Tuple, Union, Mapping
import hydra

from omegaconf import DictConfig, open_dict
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification.accuracy import (
    Accuracy, 
    MulticlassAccuracy, 
    BinaryAccuracy
)
from torchmetrics.wrappers import MetricTracker

def create_metrics_collection(
        num_classes: int,
        metrics_config: DictConfig, 
        valid: bool = False
    ) -> Union[MetricCollection, MetricTracker]:
    """
    Create a collection of metrics to track. Always include the accuracy.
    For validation, also create a metric tracker to track the best metric state.
    """

    # copy the config to avoid modifying the original
    copy_config = copy.deepcopy(metrics_config)
    # always report the accuracy
    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes, average="micro"),
    }
    min_or_max = [True]

    for key, value in copy_config.items():
        # remove the keyword if max from the value 
        # then we use that in the metric tracker
        with open_dict(value):
            min_or_max.append(value.pop("maximize"))
        metrics_dict[key] = hydra.utils.instantiate(value)
        
    metrics_collection = MetricCollection(metrics_dict)
    if valid:
        tracker = MetricTracker(metrics_collection, maximize=min_or_max)
        return tracker
    
    return metrics_collection