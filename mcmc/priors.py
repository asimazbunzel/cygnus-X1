"""Prior distributions for different stellar parameters
"""

from typing import Any, Dict, List, Union

import numpy as np
import scipy.stats as stats


def lg_prior_porb(
    porb: Union[float, List[float]],
    porb_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Prior distribution on orbital period (Porb) in logarithm

    Parameters
    ----------
    porb : `Union[float, List[float]]`
        Orbital period in days where log(PDF) will be computed

    porb_fixed : `float`
        Orbital period value of Cygnus X-1 in days

    distribution : `str`
        Name of statistical distribution to be used to compute log(PDF). Accepts any value from
        `scipy.stats`

    kwargs: `dict`
        Valid arguments for distributions found in `scipy.stats`. E.g., `loc`, `scale`

    Returns
    -------
    logpdf: `float`
        Result of the difference: log(PDF_porb) - log(PDF_porb_fixed)
    """

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [PORB - PORB_ERR , PORB + PORB_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(porb, loc=loc, scale=scale) - distro.logpdf(
            porb_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf


def lg_prior_ecc(
    ecc: Union[float, List[float]],
    ecc_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Prior distribution on eccentricity in logarithm"""

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [ECC - ECC_ERR , ECC + ECC_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(ecc, loc=loc, scale=scale) - distro.logpdf(
            ecc_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf


def lg_prior_m2(
    m2: Union[float, List[float]],
    m2_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Prior distribution on companion, non-degenerate, star (M2) in logarithm"""

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [M_2 - M_2_ERR , M_2 + M_2_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(m2, loc=loc, scale=scale) - distro.logpdf(
            m2_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf


def lg_prior_mbh(
    mbh: Union[float, List[float]],
    mbh_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Prior distribution on black hole (MBH) in logarithm"""

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [M_BH - M_BH_ERR , M_BH + M_BH_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(mbh, loc=loc, scale=scale) - distro.logpdf(
            mbh_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf


def lg_prior_inc(
    inc: Union[float, List[float]],
    inc_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Prior distribution on inclination of binary orbit before and after asymmetric kick"""

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [INC - INC_ERR , INC + INC_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(inc, loc=loc, scale=scale) - distro.logpdf(
            inc_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf


def lg_prior_vsys(
    vsys: Union[float, List[float]],
    vsys_fixed: float,
    distribution: str = "",
    **kwargs: float,
) -> Union[float, List[float]]:
    """Uniform distribution prior on systemic velocity (v_sys) in logarithm"""

    # load chosen distribution
    try:
        distro = stats.__dict__.get(distribution)
    except Exception as e:
        raise e

    # needed values
    loc: float = kwargs.get("loc")  # type: ignore
    scale: float = kwargs.get("scale")  # type: ignore
    # in case uniform, range is [VSYS - VSYS_ERR , VSYS + VSYS_ERR]
    if distribution == "uniform":
        loc = kwargs["loc"] - kwargs["scale"]
        scale = kwargs["loc"] + kwargs["scale"]

    if loc < 0:
        loc = 0

    try:
        logpdf = distro.logpdf(vsys, loc=loc, scale=scale) - distro.logpdf(
            vsys_fixed, loc=loc, scale=scale
        )

    except Exception as e:
        raise e

    return logpdf
