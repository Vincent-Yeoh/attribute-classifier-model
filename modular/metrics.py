from sklearn import metrics
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix, classification_report
from abc import ABC, abstractmethod
from typing import Callable, Dict
import torch
import numpy as np

class Result(ABC):
    def __init__(self, loss=None):
        self.loss = loss
    
    @abstractmethod
    def __str__(self):
        return f"Loss: {loss:3f}"
    
    
class MultiLabelResult(Result):
    def __init__(self, report, loss):
        super().__init__(loss=loss)
        self.report = report
     
    def __str__(self):

        return f"{self.report}"


def label_wise_accuracy(output, target):
    pred = (output > 0.5).float()
    correct = (pred == target).float()
    label_accuracy = torch.mean(correct)
    return label_accuracy.item()

def hamming_loss_fn(y_true,y_pred):
    y_pred, y_true = y_pred.detach().cpu(), y_true.detach().cpu()   
    y_pred_normalized = (y_pred > 0.5).float()  
    loss = hamming_loss(y_true=y_true, y_pred=y_pred_normalized)
    return loss


def normalize_tensor(data:torch.Tensor, threshold:float):
    data = torch.sigmoid(input=data)
    return (data>threshold).float()

def classification_report_fn(y_true, y_pred, classes):
    y_pred = normalize_tensor(data=y_pred, threshold=0.5)
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    return classification_report(y_true=y_true,
                      y_pred=y_pred,
                      output_dict=True,
                      target_names=classes,
                      zero_division=0.0
                      )
                      
def lrap_fn(y_true: torch.tensor, y_pred: torch.tensor) -> float:
    
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    # print(f'y_true: {y_true} | y_pred : {y_pred}')
    label_ranking_average_precision = metrics.label_ranking_average_precision_score(y_true=y_true, y_score=y_pred)
    return label_ranking_average_precision

def report_avg(dict_list: list):
    matrix_o = np.array([[[x for x in value.values()] for key,value in batch.items()] for batch in dict_list])
    matrix = matrix_o.mean(axis=0)
    avg = {}
    for i,(label, value) in list(enumerate(dict_list[0].items())):
        avg[label] = {'precision': matrix[i][0], 'recall': matrix[i][1], 'f1-score': matrix[i][2], 'support': matrix[i][3]}
    return avg

def multiclass_accuracy_fn(y_true, y_pred_logits):
    y_pred = y_pred_logits.argmax(dim=1)
    return (y_pred == y_true).sum().item()/len(y_pred)
