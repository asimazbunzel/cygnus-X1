# Cygnus X-1: Markov Chain Monte Carlo computation

Module to search for stellar parameters on the High-mass X-ray binary (HMXB) Cygnus X-1

## Installation

The module comes with an environment `environment.yml`. Install it with the `Makefile`, by running
`make environment`. After this is installed, all libraries needed to run the code will be available
in a conda environment called `cygnusx1`. Before running the code, make sure to activate it with
`conda activate cygnusx1`

## Running the code

Options for the MCMC exploration are located in the `config.yml` file. Change it as you wish. Once
this is done, the code can be run with `make run`

#### Notes on priors

Prior distributions are defined in the `priorDistributions` options inside the `config.yml` file.
Available options are the same as all the distributions found in the `scipy.stats` module. The
`loc` and `scale` values are set with the different values in the `CygnusX1` options. For example,
in the case of a gaussian distribution (`"norm"` in `scipy.stats`) for the orbital period (`PORB`),
`loc = PORB` and `scale = PORB_ERR`. On the contrary, for a uniform distribution (`uniform` in
`scipy.stats`) `loc = PORB - PORB_ERR` and `scale = PORB + PORB_ERR`

## Developing

The `Makefile` contains commands for development of the code. The environment created via the
`environment.yml` file will be updated with new modules after running `make dev-install` which
will install `black`, `isort`, `mypy` and `pyupgrade`

After changing stuff in the code, run `make lint` and check if the code passes all tests. Only
after it does, push changes to the remote repository
