import numpy as np
import poskiorb

import priors

M_BH = 20
PORB = 5.6
M_2 = 40e0
M_2_err = 7e0

def log_likelihood(args):

    porb_pre = args[0]
    m1_pre = args[1]
    m2 = args[2]
    w = args[3]
    theta = args[4]
    phi = args[5]

    # posterior on m1_pre, kick, porb_pre
    if m1_pre < M_BH or w < 0e0 or w > 50e0 or porb_pre < 0e0:
        return -np.inf

    # angles
    if theta < 0 or theta >= np.pi or phi < 0 or phi >= 2 * np.pi:
        return -np.inf

    # prior on companion mass
    if m2 < 20 or m2 > 40:
        return -np.inf

    # convert Period to Separation
    a_pre = poskiorb.utils.P_to_a(period=porb_pre, m1=m1_pre, m2=m2)

    # evaluate kicks model
    a_post, P_post, e, cos_i, v_sys, _, _, _, _ = poskiorb.utils.binary_orbits_after_kick(a=a_pre, m1=m1_pre, m2=m2, m1_remnant_mass=M_BH, w=w, theta=theta, phi=phi, ids=np.ones(1))
    
    # inclination limits
    inc = np.arccos(cos_i)
    if inc < 10 or inc > 50:
        return -np.inf

    # prior on v_sys
    if v_sys < 0 or v_sys > 500:
        return -np.inf

    if e < 0 or e >= 1 or a_post < 0:
        return -np.inf
    
    ll = priors.lg_norm_prior_Porb(x=P_post, porb=PORB)
    ll += priors.lg_unif_prior_ecc(x=e)
    ll += priors.lg_norm_prior_M2(x=m2, m2=M_2) 
    ll += priors.lg_unif_prior_vsys(x=v_sys)

    # prior on theta
    lp = np.log(np.sin(theta))

    return ll + lp
