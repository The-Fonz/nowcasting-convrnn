import numpy as np


def calc_metrics(t, p):
    """
    Calculates the Critical Success Index of tresholded 0/1 matrix
    :param t: np.array ground truth
    :param p: prediction
    :returns: Tuple of (CSI, FAR, POD, correlation) normalized scores
    """
    hits = ((t==1)&(p==1)).sum()
    misses = ((t==1)&(p==0)).sum()
    falsealarms = ((t==0)&(p==1))
    
    # Normalize scores by dividing by total # of pixels    
    csi = hits / (hits + misses + falsealarms) / t.size()
    far = falsealarms / (hits + falsealarms) / t.size()
    pod = hits / (hits + misses) / t.size()
    
    # Correlation
    cor = (t*p).sum() / (np.sqrt((p**2).sum() * (t**2).sum()) + 10e-9)
    
    return csi, far, pod, cor
