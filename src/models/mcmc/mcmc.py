"""Markov Chain Montecarlo calculation of stellar parameters of Cygnus X-1
"""

from typing import Any, Union

import argparse
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import emcee
import likelihood
import numpy as np
import yaml

# print options
np.set_printoptions(precision=4)

# hide warnings
warnings.filterwarnings("ignore")


desc = """ Monte Carlo evaluation of stellar parameters of the HMXB Cygnus X-1, using Markov
chain approach based on the `emcee` python module
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=desc,
        epilog="@asimazbunzel on GitHub",
    )
    parser.add_argument(
        "-C",
        "--config-file",
        dest="config_file",
        help="path to configuration file in YAML format",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        dest="debug",
        help="enable debug mode",
    )

    return parser.parse_args()


# logging stuff
def set_logger(debug: bool = False):
    """Set logging stuff"""

    # enable debug if requested
    level = logging.INFO
    if debug:
        level = logging.DEBUG

    logging.basicConfig(
        filename=".mcmc.log",
        filemode="w",
        format="%(asctime)s -- %(levelname)s -- %(message)s (%(funcName)s in %(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
        level=level,
    )

    logger = logging.getLogger("MCMC")

    return logger


def load_yaml(fname: Union[str, Path]) -> Any:
    """Load configuration file with YAML format

    Parameters
    ----------
    fname : `str / Path`
        YAML filename

    Returns
    -------
    `yaml.load`
    """

    if isinstance(fname, Path):
        fname = str(fname)

    with open(fname) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main(config_file: str = "") -> None:
    """Main driver of MCMC chain evaluation"""

    logger.info("setting Markov Chain Monte Carlo simulation")

    # load configuration
    config = load_yaml(fname=config_file)

    # set some constant values
    nwalkers = config["MCMC"].get("walkers")
    ndim = config["MCMC"].get("dimension")
    nsteps = config["MCMC"].get("steps")
    use_rand_uniform = config["MCMC"].get("use_random_uniform_walkers")
    progress = config["MCMC"].get("progress_bar")
    priors = config["MCMC"].get("priorDistributions")
    filename = config["MCMC"].get("filename")

    # Cygnus X-1 properties
    stellarParameters = config["StellarParameters"]

    # initial guess for parameter values
    # [p_pre   m1_pre    m2    w      theta     phi]
    initialGuess = config["MCMC"].get("initialGuess")
    initial_values = [
        initialGuess.get("porb_preSN"),
        initialGuess.get("m1_preSN"),
        initialGuess.get("m2"),
        initialGuess.get("w"),
        initialGuess.get("theta"),
        initialGuess.get("phi"),
    ]

    # add some randomness to initial values
    if use_rand_uniform:
        rng_lo = initialGuess.get("porb_preSN_lo") - initial_values[0]
        rng_hi = initialGuess.get("porb_preSN_hi") - initial_values[0]
        porb_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        rng_lo = initialGuess.get("m1_preSN_lo") - initial_values[1]
        rng_hi = initialGuess.get("m1_preSN_hi") - initial_values[1]
        m1_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        rng_lo = initialGuess.get("m2_lo") - initial_values[2]
        rng_hi = initialGuess.get("m2_hi") - initial_values[2]
        m2_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        rng_lo = initialGuess.get("w_lo") - initial_values[3]
        rng_hi = initialGuess.get("w_hi") - initial_values[3]
        w_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        rng_lo = initialGuess.get("theta_lo") - initial_values[4]
        rng_hi = initialGuess.get("theta_hi") - initial_values[4]
        theta_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        rng_lo = initialGuess.get("phi_lo") - initial_values[5]
        rng_hi = initialGuess.get("phi_hi") - initial_values[5]
        phi_rng = np.random.uniform(rng_lo, rng_hi, nwalkers)

        randomness = np.column_stack((porb_rng, m1_rng, m2_rng, w_rng, theta_rng, phi_rng))
    else:
        logger.critical("`use_random_uniform_walkers` = False is not yet supported")
        sys.exit(1)

    # initial walkers
    logging.debug("Initial walkers")
    initial = initial_values + randomness
    for k, el in enumerate(initial):
        logging.debug(f"walker {k}: {el}")

    # need a numpy array to start emcee
    initial = np.array(initial)

    # output handling (backend emcee)
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # update kwargs dict with info regarding priors
    kwargs = dict()
    kwargs.update(stellarParameters)
    kwargs.update(priors)

    print("starting Monte Carlo simulation")
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=likelihood.log_likelihood,
            pool=pool,
            backend=backend,
            kwargs=kwargs,
        )

        # run MCMC
        sampler.run_mcmc(initial, nsteps, progress=progress)


if __name__ == "__main__":
    args = parse_args()

    logger = set_logger(args.debug)

    logger.info("********************************************************")
    logger.info("         Markov Chain Monte Carlo calculator            ")
    logger.info("********************************************************")

    # time it
    _startTime = time.time()

    main(config_file=args.config_file)

    # time it
    _endTime = time.time()

    logger.info(f"[-- manager uptime: {_endTime - _startTime:.2f} sec --]")
