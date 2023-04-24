run:
	python mcmc/mcmc.py

codecheck:
	black mcmc/*.py

environment:
	conda env create -f environment.yml
