import numpy as np
import sklearn.metrics


def auc_var(y_true, y_score, n_iter):
    """
    from:
    https://cs.stanford.edu/people/ihaque/posters/EC4-AUC_Confidence_Intervals.pdf
    http://papers.nips.cc/paper/2645-confidence-intervals-for-the-area-under-the-roc-curve.pdf
    """
    y_true = y_true.ravel().astype(np.bool)
    y_score = y_score.ravel()
    A = sklearn.metrics.roc_auc_score(y_true, y_score)
    m = y_true.sum()
    n = len(y_true) - m
    A2 = A ** 2
    pos = y_score[y_true]
    neg = y_score[~y_true]
    N_xxy = 0
    for _ in range(n_iter):
        x1 = pos[np.random.randint(len(pos))]
        x2 = pos[np.random.randint(len(pos))]
        y1 = neg[np.random.randint(len(neg))]
        if y1 < min(x1, x2):
            N_xxy += 1
    P_xxy = N_xxy / float(n_iter)
    N_xyy = 0
    for _ in range(n_iter):
        x1 = pos[np.random.randint(len(pos))]
        y1 = neg[np.random.randint(len(neg))]
        y2 = neg[np.random.randint(len(neg))]
        if x1 > max(y1, y2):
            N_xyy += 1
    P_xyy = N_xyy / float(n_iter)
    numerator = (
        A * (1 - A) + (m - 1) * (P_xxy - A2) + (n - 1) * (P_xyy - A2))
    var = numerator / (m * n)
    return A, var
