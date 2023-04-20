from multiprocessing import Pool

import emcee
import numpy as np
import poskiorb

import likelihood

def main() -> int:

    print("setting Markow Chain Monte Carlo simulation")

    nwalkers, ndim = (32,6)

    #                 p_pre   m1_pre    m2    w      theta     phi
    initial_values = [ 3e0,    25e0, 40e0, 10e0,  np.pi/2, np.pi/2]  # initial guess for parameter values
    initial = initial_values + np.random.normal(0,1,size=(nwalkers,ndim))/10

    #nsteps,burn = 1000,100
    nsteps,burn = 50000,20000

    filename = "cygnusx1.h5"

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    print('starting Monte Carlo simulation')
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.log_likelihood, pool=pool, backend=backend)
        sampler.run_mcmc(initial, nsteps, progress=True)

    # reshape sample
    data = sampler.get_chain()[burn:,:,:].reshape(nwalkers*(nsteps-burn),ndim)
    data=data[np.random.choice(data.shape[0], 100000, replace=False), :]

    return 0

if __name__ == "__main__":
    main()
