
.PHONY: environment
environment:
	conda env create -f environment.yml

run:
	python mcmc/mcmc.py

.PHONY: plots
plots:
	python plotter/plot.py

# -----------------------------------------------------------------------
#  stuff for development
# -----------------------------------------------------------------------
.PHONY: dev-install
dev-install:
	pip install -U black==23.3.0 flake8==6.0.0 isort==5.12.0 mypy==1.2.0 pyupgrade==3.3.2

codestyle:
	pyupgrade --exit-zero-even-if-changed --py37-plus **/*.py
	isort --settings-file pyproject.toml ./
	black --config pyproject.toml ./

mypy:
	mypy --config-file pyproject.toml ./

.PHONY: lint
lint: codestyle mypy
