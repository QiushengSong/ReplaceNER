from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Union, List
import torch
import numpy as np
from dataclasses import dataclass




@dataclass
class Evaluator:
    precision_score = None
    recall_score = None
    f1_score = None


class Evaluate(Evaluator):

    def __call__(
            self,
            total_predicted_label: Union[List, np.ndarray, torch.Tensor],
            total_original_label: Union[List, np.ndarray, torch.Tensor]
            ) -> float:

        total_predicted_label = total_predicted_label.numpy()

        total_original_label = total_original_label.numpy()

        self.precision_score = precision_score(total_original_label, total_predicted_label, average='micro')

        self.recall_score = recall_score(total_original_label, total_predicted_label, average='micro')

        self.f1_score = f1_score(total_original_label, total_predicted_label, average='micro')

        return self.precision_score, self.recall_score, self.f1_score
