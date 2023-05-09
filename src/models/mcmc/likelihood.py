"""likelihood

Compute log of likelihood while imposing conditions measured in Cygnus X-1
"""

from typing import List

import logging

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

    # check angles
    phi = phi % (2 * np.pi)
    theta = theta % (2 * np.pi)
    if theta < 0 or theta > np.pi:
        theta = np.pi - (theta % np.pi)
    
    # this is for debugging purposes
    binary_str = f"m1 = {m1_pre:.2f}, m2 = {m2:.2f}, p = {porb_pre:.2e} :: "
    binary_str += f"ω = {w:.2e}, θ = {np.rad2deg(theta):.2e}, φ = {np.rad2deg(phi):.2e}"

    # remove cases that are not physical
    if m1_pre < float(kwargs["M_BH"]):
        logger.debug(f"{binary_str} :: non physical case (m1 < {kwargs['M_BH']:.2f})")
        return -np.inf
    if w < 0e0:
        logger.debug(f"{binary_str} :: non physical case (ω < 0)")
        return -np.inf
    if porb_pre < 0e0:
        logger.debug(f"{binary_str} :: non physical case: (p < 0)")
        return -np.inf
    if theta < 0 or theta >= np.pi or phi < 0 or phi >= 2 * np.pi:
        logger.debug(f"{binary_str} :: angles outside limits: (θ = {theta:.2f}, φ = {phi:.2f})")
        return -np.inf

    # remove unlikely scenarios
    if porb_pre > 1e3:
        logger.debug(f"{binary_str} :: unlikely scenario: (p > 1000)")
        return -np.inf
    if w > 600e0:
        logger.debug(f"{binary_str} :: unlikely scenario: (w > 500)")
        return -np.inf
    if m2 < (kwargs["M_2"] - kwargs["M_2_ERR"]) or m2 > (kwargs["M_2"] + kwargs["M_2_ERR"]):
        logger.debug(
            f"{binary_str} :: unlikely scenario: "
            f"(m2 < {kwargs['M_2'] - kwargs['M_2_ERR']} | m2 > {kwargs['M_2'] + kwargs['M_2_ERR']})"
        )
        return -np.inf

    # convert period to separation, needed for the `binary_orbits_after_kick`
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
    
    # we dont want unbounded binaries
    if not np.isfinite(e):        
        logger.debug(f"{binary_str} :: unbounded after kick")
        return -np.inf

    # inclination to deg.
    inc = np.rad2deg(np.arccos(cos_i))

    # compute priors to update likelihood
    try:
        log_L: float = priors.lg_prior_porb(
            porb=p_post,
            porb_fixed=kwargs["PORB"],
            distribution=kwargs["p_orb"],  # type: ignore
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
        sys.exit()

    except Exception as exc:
        logger.critical(f"could not compute log_L: {str(exc)}")
        sys.exit()

    # prior on theta, phi => isotropic distribution pdf = 0.5 * sin(θ)
    log_L += np.log(np.sin(theta))

    # debugging stuff
    if log_L != -np.inf:
        logger.debug(
            f"{binary_str} :: "
            f"P = {p_post:.2e}, e = {e:.2f}, i = {inc:.2e}, v_sys = {v_sys:.2e} => "
            f"log_L = {log_L:.2f}"
        )

    return log_L
