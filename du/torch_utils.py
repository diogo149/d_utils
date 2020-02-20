import du
import collections
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def adjust_optimizer_param(optimizer, k, v):
    """
    adjust parameter of an optimizer e.g. lr
    """
    for param_group in optimizer.param_groups:
        param_group[k] = v


def apply_to_optimizer_param(optimizer, k, fn):
    """
    apply a function to an optimizer's parameter
    """
    for param_group in optimizer.param_groups:
        param_group[k] = fn(param_group[k])


def adjust_learning_rate(optimizer, lr):
    """Adjust the learning rate of an optimizer"""
    adjust_optimizer_param(optimizer, "lr", lr)


def save_model(trial, name, model):
    with du.timer(
        "save model (%s) for %s:%d" % (name, trial.trial_name, trial.iteration_num)
    ):
        torch.save(model.state_dict(), trial.file_path("model_%s.pth" % name))


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


def random_seed(seed):
    """
    randomly seed relevant values
    see: https://discuss.pytorch.org/t/what-is-manual-seed/5939/16

    note: this doesn't make everything determinisitc, because
    CuDNN may not be:
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_entropy_loss(logits, probs, axis=-1, reduction="mean"):
    """
    cross-entropy loss that supports probabilities as floats

    TODO is this the same as nn.KLDivLoss()
    """
    tmp = -torch.sum(probs * F.log_softmax(logits, dim=axis), axis=axis)
    if reduction == "mean":
        return tmp.mean()
    else:
        raise ValueError


def label_smoothing(target, epsilon, num_classes=-1):
    if target.ndim == 1 and target.dtype == torch.int64:
        target = F.one_hot(target, num_classes=num_classes).float()
    num_classes = target.shape[-1]
    uniform_weight = epsilon / (num_classes - 1)
    target_scaling = 1.0 - epsilon - uniform_weight
    uniform = torch.ones_like(target) * uniform_weight
    return target * target_scaling + uniform


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, idx):
        # assumes all datasets use integer indices
        # warning: no error checking
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx]) and dataset_idx < (
            len(self.datasets) - 1
        ):
            idx -= len(self.datasets[dataset_idx])
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmented_data):
        assert len(dataset) == len(augmented_data)
        self.dataset = dataset
        self.augmented_data = augmented_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx] + (self.augmented_data[idx],)


def cifar10_standard_augmentation():
    return [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]


def cifar10_transforms():
    return [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]


def cifar10_data(*, train, augmentation=None, transform_list=None):
    """
    train: whether or not to access the train set

    augmentation:
      "standard": translation + h flipping
      None: no augmentation
    """
    if transform_list is None:
        transform_list = cifar10_transforms()
        if augmentation is None:
            # do nothing
            pass
        elif augmentation == "standard":
            transform_list = cifar10_standard_augmentation() + transform_list
        elif isinstance(augmentation, list):
            transform_list = augmentation + transform_list
        else:
            raise ValueError("invalid augmentation %s" % repr(augmentation))

    transform = transforms.Compose(transform_list)

    return datasets.CIFAR10(
        root="~/data", train=train, download=True, transform=transform
    )


def mnist_transforms():
    return [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]


def mnist_data(*, train, augmentation=None, transform_list=None):
    if transform_list is None:
        transform_list = mnist_transforms()

        if augmentation is None:
            # do nothing
            pass
        elif isinstance(augmentation, list):
            transform_list = augmentation + transform_list
        else:
            raise ValueError("invalid augmentation %s" % repr(augmentation))

    transform = transforms.Compose(transform_list)
    return datasets.MNIST(
        root="~/data", train=train, download=True, transform=transform
    )


def device_chooser(device_str, cudnn_benchmark=True):
    if device_str is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = cudnn_benchmark
            # enable multi-gpu
            # FIXME model isn't given
            # raise NotImplementedError
            # if torch.cuda.device_count() > 1:
            #     # model = torch.nn.DataParallel(model)
            #     pass
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    return device


def copy_state_dict(state_dict, device=None):
    new_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        # on best approach to copying tensors:
        # https://stackoverflow.com/questions/49178967/copying-a-pytorch-variable-to-a-numpy-array
        # https://pytorch.org/docs/stable/tensors.html
        new_v = v.clone().detach()
        if device is not None:
            new_v = new_v.to(device)
        new_dict[k] = new_v
    return new_dict


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    NOTE: meant to be places after transforms.ToTensor()

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.

    from https://github.com/uoguelph-mlrg/Cutout
    """

    def __init__(self, n_holes=1, length=0.5):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if isinstance(self.length, int):
            h_len = w_len = self.length
        elif isinstance(self.length, float):
            h_len = int(round(h * self.length))
            w_len = int(round(w * self.length))
        else:
            raise ValueError(self.length)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - h_len // 2, 0, h)
            y2 = np.clip(y + h_len // 2, 0, h)
            x1 = np.clip(x - w_len // 2, 0, w)
            x2 = np.clip(x + w_len // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
