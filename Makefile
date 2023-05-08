
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = cygnus-X1

# set up python interpreter environment
environment:
	@echo "creating environment for $(PROJECT_NAME), located in $(PROJECT_DIR)"
	conda env create -f config/environment.yml

# rules to run MCMC code & helpers
.PHONY: mcmc-chain, mcmc-help, process-data
mcmc-chain:
	python src/models/mcmc/mcmc.py --config-file $(PROJECT_DIR)/config/mcmc-config.yml

mcmc-help:
	python src/models/mcmc/mcmc.py --help

process-data:
	python src/features/clean_chain.py --config-file $(PROJECT_DIR)/config/mcmc-config.yml

## delete all compiled python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# install jupyter to open jupyter notebook
install-jupyter:
	conda install -c conda-forge jupyter

# development stuff
.PHONY: dev
dev-environment:
	pip install -U black==23.3.0 flake8==6.0.0 isort==5.12.0 mypy==1.2.0 pyupgrade==3.3.2

# linting
.PHONY: codestyle
codestyle:
	pyupgrade --exit-zero-even-if-changed --py37-plus **/*.py
	isort --settings-file config/pyproject.toml ./
	black --config config/pyproject.toml ./
	flake8 ./ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ./ --count --exit-zero --max-complexity=12 --max-line-length=127 --statistics 
