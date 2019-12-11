import du
import torch


def adjust_learning_rate(optimizer, lr):
    """Adjust the learning rate of an optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(trial, name, model):
    with du.timer("save model (%s) for %s:%d" %
                  (name, trial.trial_name, trial.iteration_num)):
        torch.save(model.state_dict(),
                   trial.file_path("model_%s.pth" % name))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
