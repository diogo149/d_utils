import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import du
import du.torch_utils
from du._test_utils import equal, numpy_allclose


def test_cross_entropy_loss():
    logits = torch.tensor(np.random.randn(4, 3))
    targets = torch.tensor([0, 1, 2, 1])
    probs = F.one_hot(targets, num_classes=3)
    ans = nn.CrossEntropyLoss()(logits, targets)
    numpy_allclose(
        du.torch_utils.cross_entropy_loss(logits, probs),
        ans)
