"""Markov Chain Montecarlo calculation of stellar parameters of Cygnus X-1"""

import logging
from multiprocessing import Pool
from pathlib import Path
import pprint
import sys
import time
from typing import Any, Union
import warnings

import emcee
import numpy as np
import poskiorb
import yaml

import likelihood

# print options
np.set_printoptions(precision=4)

# hide warnings
warnings.filterwarnings("ignore")

# logging stuff
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler that logs debug and higher level messages
fh = logging.FileHandler(f"mcmc.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# filename with configuration
CONFIG_FILENAME = "config.yml"


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


def main() -> None:
    """Main driver of MCMC chain evaluation"""

    logger.info("setting Markov Chain Monte Carlo simulation")

    # load configuration
    config = load_yaml(fname=CONFIG_FILENAME)

    # set some constant values
    nwalkers = config["MCMC"].get("walkers")
    ndim = config["MCMC"].get("dimension")
    nburn = config["MCMC"].get("burn")
    nsteps = config["MCMC"].get("steps")
    filename = config["IO"].get("filename")

    # TODO: re-write to be used as a parameter from a YAML file
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
    use_rand_uniform = config["MCMC"].get("use_random_uniform_walkers")
    if use_rand_uniform:
        randomness = [
            np.array(
                [
                    np.random.uniform(-3, 6),
                    np.random.uniform(-4, 15),
                    np.random.uniform(-7, 7),
                    np.random.uniform(-9, 50),
                    np.random.uniform(-np.pi / 2, 3 * np.pi / 2),
                    np.random.uniform(-np.pi / 2, np.pi / 2),
                ]
            )
            for i in range(nwalkers)
        ]
    else:
        logger.critical("`use_random_uniform_walkers` = False is not yet supported")
        sys.exit(1)

    initial = []
    for k, element in enumerate(randomness):
        el = list(element)
        initial.append(element + initial_values)

    logging.debug("Initial walkers")
    for k, el in enumerate(initial):
        logging.debug(f"walker {k}: {el}")

    return

    # need a numpy array to start emcee
    initial = np.array(initial)

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    print("starting Monte Carlo simulation")
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, likelihood.log_likelihood, pool=pool, backend=backend
        )
        sampler.run_mcmc(initial, nsteps, progress=True)


if __name__ == "__main__":
    logger.info("********************************************************")
    logger.info("         Markov Chain Monte Carlo calculator            ")
    logger.info("********************************************************")

    # time it
    _startTime = time.time()

    main()

    # time it
    _endTime = time.time()

    logger.info(f"[-- manager uptime: {_endTime - _startTime:.2f} sec --]")
