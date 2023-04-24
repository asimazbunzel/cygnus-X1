from typing import List, Union

import numpy as np
import scipy.stats as stats

M_BH = 20
PORB = 5.6
ECC = 0.018
M_2 = 40e0
M_2_err = 7e0


def lg_prior_Porb(
    x: Union[float, List[float]],
    porb: Union[float, List[float]],
    distribution: str = "gaussian",
) -> Union[float, List[float]]:
    """Prior distribution on orbital period (Porb) in logarithm"""

    if distribution == "gaussian":
        return stats.norm.logpdf(x, loc=porb, scale=0.20 * porb) - stats.norm.logpdf(
            PORB, loc=porb, scale=0.20 * porb
        )
    else:
        sys.exit(
            "need to modify `lg_prior_Porb` before running this prior distribution"
        )


def lg_prior_ecc(
    x: Union[float, List[float]],
    ecc: Union[float, List[float]],
    distribution: str = "uniform",
) -> Union[float, List[float]]:
    """Prior distribution on eccentricity in logarithm"""

    if distribution == "uniform":
        return stats.uniform.logpdf(x, loc=0, scale=0.2) - stats.uniform.logpdf(
            0.1, loc=0, scale=0.2
        )
    else:
        sys.exit("need to modify `lg_prior_ecc` before running this prior distribution")


def lg_prior_M2(
    x: Union[float, List[float]],
    m2: Union[float, List[float]],
    distribution: str = "gaussian",
) -> Union[float, List[float]]:
    """Prior distribution on companion, non-degenerate, star (M2) in logarithm"""

    if distribution == "gaussian":
        return stats.norm.logpdf(x, loc=m2, scale=0.20 * m2) - stats.norm.logpdf(
            M_2, loc=m2, scale=0.20 * m2
        )
    else:
        sys.exit("need to modify `lg_prior_M2` before running this prior distribution")


def lg_prior_BH(
    x: Union[float, List[float]], mbh: Union[float, List[float]], distribution: str = ""
) -> Union[float, List[float]]:
    """Prior distribution on black hole (MBH) in logarithm"""
    return None


def lg_prior_inc(
    x: Union[float, List[float]],
    inc: Union[float, List[float]],
    distribution: str = "uniform",
) -> Union[float, List[float]]:
    """Prior distribution on inclination of binary orbit before and after asymmetric kick"""

    if distribution == "uniform":
        return stats.uniform.logpdf(x, loc=10, scale=30) - stats.norm.logpdf(
            23, loc=10, scale=30
        )
    else:
        sys.exit("need to modify `lg_prior_inc` before running this prior distribution")


def lg_prior_vsys(
    x: Union[float, List[float]],
    vsys: Union[float, List[float]],
    distribution: str = "uniform",
) -> Union[float, List[float]]:
    """Uniform distribution prior on systemic velocity (v_sys) in logarithm"""

    if distribution == "uniform":
        return stats.uniform.logpdf(x, loc=5, scale=35) - stats.uniform.logpdf(
            15, loc=5, scale=35
        )
    else:
        sys.exit(
            "need to modify `lg_prior_vsys` before running this prior distribution"
        )
