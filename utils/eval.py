from __future__ import print_function, absolute_import
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def ddd_accuracy(output, target):
    """
    Computes the precision for NTHU-DDD dataset
    """
    #print("target: ", target)
    batch_size = target.size(0)

    #_, pred = output.max(1)
    pred = output.argmax(dim=1)
    pred = pred.t()
    correct = pred.eq(target.view_as(pred)).sum()
    res = correct.mul_(100.0 / batch_size)
    return res



def ssd_accuracy(output, target):
    """
    Computes the precision for PASCAL VOC dataset
    """
    batch_size = target.size(0)
    object_num = target.size(1)
    total_size = batch_size * object_num
    pred = output.argmax(dim=2)

    correct = pred.eq(target.view_as(pred)).float().sum()
    res = correct.mul_(100.0 / total_size)
    return res
