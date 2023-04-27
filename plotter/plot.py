"""MCMC plotter
"""

from typing import Any, Union

from pathlib import Path

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import poskiorb
import yaml

# ===================================
# filename with configuration options
CONFIG_FILENAME = "config.yml"
# ===================================


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


if __name__ == "__main__":
    # load configuration
    config = load_yaml(fname=CONFIG_FILENAME)

    # set some constant values
    nwalkers = config["MCMC"].get("walkers")
    ndim = config["MCMC"].get("dimension")
    nburn = config["MCMC"].get("burn")
    nsteps = config["MCMC"].get("steps")
    filename = config["MCMC"].get("filename")

    # Cygnus X-1 properties
    cygnusX1 = config["CygnusX1"]

    # load samples
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    samples = reader.get_chain(discard=nburn, flat=True)

    # corner plot of pre core-collapse parameters
    figure = corner.corner(
        samples,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.90),
        labels=[
            "$P_{\\rm pre}$",
            "$M_{\\rm 1, pre}$",
            "$M_2$",
            "$w$",
            "$\\theta$",
            "$\\phi$",
        ],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.savefig("output/pre_cc.png")

    # evaluate kicks model
    (
        a_post,
        p_post,
        e,
        cos_i,
        v_sys,
        _,
        _,
        _,
        _,
    ) = poskiorb.utils.binary_orbits_after_kick(
        a=poskiorb.utils.P_to_a(samples[:, 0], samples[:, 1], samples[:, 2]),
        m1=samples[:, 1],
        m2=samples[:, 2],
        m1_remnant_mass=21,
        w=samples[:, 3],
        theta=samples[:, 4],
        phi=samples[:, 5],
        ids=np.ones(len(samples[:, 0])),
    )

    samples2 = []
    for k in range(len(a_post)):
        samples2.append([p_post[k], e[k], np.rad2deg(np.arccos(cos_i[k])), v_sys[k]])
    samples_post = np.asarray(samples2)

    # corner plot of post core-collapse parameters
    figure = corner.corner(
        samples_post,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.90),
        labels=[
            "$P_{\\rm post}$",
            "$e_{\\rm post}$",
            "$i$",
            "$v_{\\rm sys}$",
        ],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    plt.savefig("output/post_cc.png")
