import numpy as np


def calc_metrics(t, p, treshold=.5):
    """
    Calculates the Critical Success Index of mm/h rain values.
    See arxiv.org/pdf/1506.04214.pdf
    :param t: np.array ground truth
    :param p: prediction
    :kwarg treshold: Treshold at which to convert to 0/1
    :returns: Tuple of (CSI, FAR, POD, correlation) normalized scores
    """
    hits = ((t>treshold)&(p>treshold)).sum()
    misses = ((t>treshold)&(p<treshold)).sum()
    falsealarms = ((t<treshold)&(p>treshold)).sum()
    
    # Normalize scores by dividing by total # of pixels    
    csi = hits / (hits + misses + falsealarms)
    far = falsealarms / (hits + falsealarms)
    pod = hits / (hits + misses)
    
    # Correlation
    cor = (t*p).sum() / (np.sqrt((p**2).sum() * (t**2).sum()) + 10e-9)
    
    return csi, far, pod, cor


def pixelval_to_mmh(arr):
    """
    Convert pixel value to rain flux in kg/m2/h
    http://adaguc.knmi.nl/contents/datasets/productdescriptions/W_ADAGUC_Product_description_RADNL_OPER_R___25PCPRR_L3.html
    """
    r = 10 ** ((arr - 109) / 32)
    return r
