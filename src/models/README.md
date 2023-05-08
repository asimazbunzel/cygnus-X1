Markov chain Monte Carlo
========================

Code to run an MCMC analysis on asymmetric kicks on Cygnus X-1

Installation
------------

The code requires some packages, packed in a conda environment file `environment.yml` located in
the `config` directory. Install it using the `Makefile`, by running `make environment`.

After this, all the libraries needed to run the code will be available in a conda environment
called `cygnusx1`. Before running the code, make sure to activate it with
`conda activate cygnusx1`.

Running the code
----------------

Options for the MCMC exploration are located in the `config.yml` file inside the `config`
directory. Change it as you wish. Once this is done, the code can be run with `make run`

Notes on priors
---------------

Prior distributions are defined in the `priorDistributions` options inside the `config.yml` file.
Available options are the same as all the distributions found in the `scipy.stats` module. The
`loc` and `scale` values are set with the different values in the `StellarParameters` options. For
example, in the case of a gaussian distribution (`"norm"` in `scipy.stats`) for the orbital period
(`PORB`), `loc = PORB` and `scale = PORB_ERR`. On the contrary, for a uniform distribution
(`uniform` in `scipy.stats`) `loc = PORB - PORB_ERR` and `scale = PORB + PORB_ERR`

Developing
----------

The `Makefile` contains commands for development of the code. The environment created via the
`environment.yml` file will be updated with new modules after running `make dev-install` which
will install `black`, `isort` and `pyupgrade`

After changing stuff in the code, run `make codestyle` and check if the code passes all tests. Only
after it does, push changes to the remote repository
