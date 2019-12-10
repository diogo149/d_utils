import torch


def adjust_learning_rate(optimizer, lr):
    """Adjust the learning rate of an optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
