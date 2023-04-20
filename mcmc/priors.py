
from typing import List, Union

import numpy as np
import scipy.stats as stats

def lg_norm_prior_Porb(x: Union[float, List[float]], porb: Union[float, List[float]]) -> Union[float, List[float]]:
    """Normal distribution prior on orbital period (Porb) in logarithm
    """
    return stats.norm.logpdf(x, loc=porb, scale=0.20 * porb) - stats.norm.logpdf(p0, loc=porb, scale=0.20*porb)

def lg_unif_prior_ecc(x: Union[float, List[float]]) -> Union[float, List[float]]:
    """Uniform distribution prior on eccentricity in logarithm
    """
    return stats.uniform.logpdf(x, loc=0, scale=0.25) - stats.uniform.logpdf(0.15, loc=0, scale=0.25)

def lg_norm_prior_M2(x: Union[float, List[float]], m2: Union[float, List[float]]) -> Union[float, List[float]]:
    """Normal distribution prior on companion, non-degenerate, star (M2) in logarithm
    """
    return stats.norm.logpdf(x, loc=m2, scale=0.20 * m2) - stats.norm.logpdf(m2, loc=m2, scale=0.20 * m2)

def lg_norm_prior_BH(x: Union[float, List[float]], mbh: Union[float, List[float]]) -> Union[float, List[float]]:
    """Normal distribution prior on black hole (MBH) in logarithm
    """
    return None

def lg_unif_prior_vsys(x: Union[float, List[float]]) -> Union[float, List[float]]:
    """Uniform distribution prior on systemic velocity (v_sys) in logarithm
    """
    return stats.uniform.logpdf(x, loc=10, scale=10) - stats.uniform.logpdf(0.15, loc=10, scale=10)
