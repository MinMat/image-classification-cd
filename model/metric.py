import torch
from pytorch_lightning.metrics.functional import auc,f1_score


def auc(outputs, target, reorder=True):
    _, pred = torch.max(outputs, dim=1)
    score = auc(pred,target,reorder=True)
    return score


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
