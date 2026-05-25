# AP50 evaluation metric.

from typing import Callable
from torch.utils.data import Dataset
from dataset import compute_iou


# Fraction of predictions with IoU >= 0.5 against the GT box.
def evaluate_ap50(predict_fn: Callable[[dict], int], dataset: Dataset) -> float:
    correct = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        pred_box = sample["proposals"][predict_fn(sample)].tolist()
        if compute_iou(pred_box, sample["gt_box"].tolist()) >= 0.5:
            correct += 1
    return correct / len(dataset) if len(dataset) > 0 else 0.0
