Cygnus X-1
==========

Module to search for stellar parameters on the High-mass X-ray binary (HMXB) Cygnus X-1

Project Organization
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make environment` or `make mcmc-chain`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── config             <- Several configuration files
    │   ├── config-mcmc.yml
    │   ├── environment.yml
    │   ├── pyproject.toml
    │   └── requirements.txt
    │
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └─── models         <- Script to create models
    │        │
    │        └── mcmc
    │            └── mcmc.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
