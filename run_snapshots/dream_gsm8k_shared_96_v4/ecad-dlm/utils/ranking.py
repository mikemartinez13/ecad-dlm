"""Rank statistics used to compare proxies with measured outcomes."""
from __future__ import annotations

import numpy as np


def rank_score(x,y):
    from scipy.stats import spearmanr
    if len(set(x))<2 or len(set(y))<2: return None
    return float(spearmanr(x,y).statistic)


def auc_score(scores,labels):
    from scipy.stats import rankdata
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores)
    n=int(labels.sum());m=len(labels)-n
    if n==0 or m==0:return None
    return float((rankdata(scores)[labels].sum()-n*(n+1)/2)/(n*m))
