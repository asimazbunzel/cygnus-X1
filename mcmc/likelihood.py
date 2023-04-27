"""likelihood

Compute log of likelihood while imposing conditions measured in Cygnus X-1
"""

from typing import Any, Dict, List

import logging
import sys

import numpy as np
import poskiorb
import priors

# logging stuff
logger = logging.getLogger()


def log_likelihood(args: List[float], **kwargs: float) -> float:
    """Compute logarithm of the likelihood

    Parameters
    ----------
    args : `List[float]`
        Array of elements to explore in MCMC

    kwargs : `dict`
        Dictionary with stellar parameters of Cygnus X-1 (see `mcmc.py` for references)
    """

    # set parameters at asymmetric kick initial moments
    porb_pre = args[0]
    m1_pre = args[1]
    m2 = args[2]
    w = args[3]
    theta = args[4]
    phi = args[5]

    # remove cases that are not physical
    if m1_pre < float(kwargs["M_BH"]):
        logger.debug(f"found non physical case: m1_pre = {m1_pre:.2f} (< {kwargs['M_BH']:.2f})")
        return -np.inf
    if w < 0e0:
        logger.debug(f"found non physical case: w = {w:.2e} (< 0)")
        return -np.inf
    if porb_pre < 0e0:
        logger.debug(f"found non physical case: porb_pre = {porb_pre:.2e} (< 0)")
        return -np.inf

    # check angles
    phi = phi % (2 * np.pi)
    if theta < 0 or theta > np.pi:
        theta = np.pi - (theta % np.pi)

    if theta < 0 or theta >= np.pi or phi < 0 or phi >= 2 * np.pi:
        logger.debug(f"found angles larger than limits: theta = {theta:.2f}, phi = {phi:.2f}")
        return -np.inf

    # convert period to separation
    a_pre = poskiorb.utils.P_to_a(period=porb_pre, m1=m1_pre, m2=m2)

    # evaluate kicks model
    (
        a_post,
        p_post,
        e,
        cos_i,
        v_sys,
        _,
        _,
        _,
        _,
    ) = poskiorb.utils.binary_orbits_after_kick(
        a=a_pre,
        m1=m1_pre,
        m2=m2,
        m1_remnant_mass=kwargs["M_BH"],
        w=w,
        theta=theta,
        phi=phi,
        ids=np.ones(1),
    )

    if len(a_post) <= 0:
        logger.debug("found non-surviving binary after kick")
        return -np.inf

    # convert arrays of 1 element to floats
    a_post = a_post[0]
    p_post = p_post[0]
    e = e[0]
    cos_i = cos_i[0]
    v_sys = v_sys[0]

    # inclination to deg.
    inc = np.rad2deg(np.arccos(cos_i))

    # we dont want unbounded binaries
    if e < 0 or e >= 1 or a_post < 0:
        logger.debug(f"found non-surviving binary after kick: e = {e:.2e}, a_post = {a_post:.2e}")
        return -np.inf

    # compute priors to update likelihood
    try:
        log_L: float = priors.lg_prior_porb(
            porb=p_post,
            porb_fixed=kwargs["PORB"],
            distribution=kwargs["porb"],  # type: ignore
            loc=kwargs["PORB"],
            scale=kwargs["PORB_ERR"],
        )
        log_L += priors.lg_prior_ecc(  # type: ignore
            ecc=e,
            ecc_fixed=kwargs["ECC"],
            distribution=kwargs["e"],  # type: ignore
            loc=kwargs["ECC"],
            scale=kwargs["ECC_ERR"],
        )
        log_L += priors.lg_prior_m2(  # type: ignore
            m2=m2,
            m2_fixed=kwargs["M_2"],
            distribution=kwargs["m2"],  # type: ignore
            loc=kwargs["M_2"],
            scale=kwargs["M_2_ERR"],
        )
        log_L += priors.lg_prior_vsys(  # type: ignore
            vsys=v_sys,
            vsys_fixed=kwargs["VSYS"],
            distribution=kwargs["v_sys"],  # type: ignore
            loc=kwargs["VSYS"],
            scale=kwargs["VSYS_ERR"],
        )
        log_L += priors.lg_prior_inc(  # type: ignore
            inc=inc,
            inc_fixed=kwargs["INC"],
            distribution=kwargs["i"],  # type: ignore
            loc=kwargs["INC"],
            scale=kwargs["INC_ERR"],
        )

    except TypeError:
        logger.critical(
            "to use more complicated `scipy.stats` distributions, need to modify "
            "`priors.py` to adapt it for such cases"
        )
        raise TypeError(
            "`scipy.stats` distribution need more complex prior. edit `priors.py` " "and try again"
        )

    except Exception as exc:
        logger.error(f"could not compute log_L: {str(exc)}")
        log_L = -np.inf

    # prior on theta
    log_L += np.log(np.sin(theta))

    # debugging stuff
    if log_L != -np.inf and np.abs(log_L) < 0.01:
        logger.debug(
            f"P = {p_post:.2e}, e = {e:.2f}, i = {inc:.2e}, v_sys = {v_sys:.2e}, "
            f"w = {w:.2e}, theta = {theta:.2f}, phi = {phi:.2f} => log_L = {log_L:.2f}"
        )

    return log_L
