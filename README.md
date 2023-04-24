# Cygnus X-1: Markov Chain Monte Carlo computation

Module to search for stellar parameters on the High-mass X-ray binary (HMXB) Cygnus X-1

## Installation

The module comes with an environment `environment.yml`. Install it with the `Makefile`,
by running `make environment`

## Running the code

Options for the MCMC exploration are located in the `config.yml` file. Change it as
you wish. Once this is done, the code can be run with `make run`

If different priors for the stellar parameters are needed, modify the `priors.py` file
in the `mcmc` module

## Developing

Styling of the code is done via the **black** module. Use it with `make codecheck`
