import sys

import numpy as np
import poskiorb

import priors

# parameters of Cygnus X-1
M_BH = 20  # Msun
M_2 = 40e0  # Msun
PORB = 5.6  # days
VSYS = 10.7  # km/s
ECC = 0.019
INC = 23  # deg

DBG = False


def log_likelihood(args):
    porb_pre = args[0]
    m1_pre = args[1]
    m2 = args[2]
    w = args[3]
    theta = args[4]
    phi = args[5]

    # posterior on m1_pre, kick, porb_pre
    if m1_pre < M_BH or w < 0e0 or w > 250e0 or porb_pre < 0e0:
        return -np.inf

    # angles
    if theta < 0 or theta >= np.pi or phi < 0 or phi >= 2 * np.pi:
        return -np.inf

    # convert Period to Separation
    a_pre = poskiorb.utils.P_to_a(period=porb_pre, m1=m1_pre, m2=m2)

    # evaluate kicks model
    (
        a_post,
        P_post,
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
        m1_remnant_mass=M_BH,
        w=w,
        theta=theta,
        phi=phi,
        ids=np.ones(1),
    )

    # convert from array to float
    try:
        a_post = a_post[0]
        P_post = P_post[0]
        e = e[0]
        cos_i = cos_i[0]
        v_sys = v_sys[0]
    except:
        e = -1
        pass

    # inclination to deg.
    inc = np.arccos(cos_i) * 180 / np.pi

    # we dont want unbounded binaries
    if e < 0 or e >= 1 or a_post < 0:
        return -np.inf

    # compute priors to update likelihood
    log_L = priors.lg_prior_Porb(x=P_post, porb=PORB)
    log_L += priors.lg_prior_ecc(x=e, ecc=ECC)
    log_L += priors.lg_prior_M2(x=m2, m2=M_2)
    log_L += priors.lg_prior_vsys(x=v_sys, vsys=VSYS)
    log_L += priors.lg_prior_inc(x=inc, inc=INC)

    # prior on theta
    log_L += np.log(np.sin(theta))

    if isinstance(log_L, np.ndarray):
        log_L = -np.inf

    # debugging stuff
    if DBG and log_L != -np.inf and inc > 16:
        print(
            f"P = {P_post:.2f}, e = {e:.2f}, i = {inc:.2f}, v_sys = {v_sys:.2f}, w = {w:.2f}, theta = {theta:.2f}, phi = {phi:.2f}: \t log_L = {log_L:.2f}"
        )

    return log_L
