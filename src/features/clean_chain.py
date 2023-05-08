"""Modify MCMC chain to burn first few steps and compute post-collapse parameters"""

from typing import Any, Union

import argparse
import sys
from pathlib import Path

import emcee
import h5py
import numpy as np
import poskiorb
import yaml

sys.path.append("src/models/mcmc")
from likelihood import log_likelihood


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="clean MCMC chain from burners & compute stellar parameters after collapse",
        epilog="@asimazbunzel on GitHub",
    )
    parser.add_argument(
        "-C",
        "--config-file",
        dest="config_file",
        help="path to configuration file in YAML format",
        type=str,
    )

    return parser.parse_args()


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
    # load config file
    config = load_yaml(fname=config_file)

    # set some constant values
    nburn = config["MCMC"].get("burn")
    filename = config["MCMC"].get("filename")
    output_filename = config["MCMC"].get("processed_filename")
    priorsD = config["MCMC"].get("priorDistributions")
    # Cygnus X-1 properties
    stellarParameters = config["StellarParameters"]

    # load samples & remove burned steps
    print("loading MCMC chain and burning steps", end="... ")
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    samples = reader.get_chain(flat=True, discard=nburn)
    # to make plots we don't need more than 10 000 samples, choose them randomly
    samples_r = samples[np.random.choice(samples.shape[0], 10000, replace=False), :]
    print("done !")

    print("samples pre-CC shape:", samples_r.shape, type(samples_r))

    kwargs = dict()
    kwargs.update(stellarParameters)
    kwargs.update(priorsD)

    # evaluate kicks model
    print("computing binary stellar parameters after kick", end="... ", flush=True)
    samples1, samples2 = [], []
    for k in range(len(samples_r)):
        # replace values of theta and phi outside of boundaries:
        # 0 < theta < pi, and 0 < phi < 2 * pi
        theta = samples_r[k, 4]
        phi = samples_r[k, 5] % (2 * np.pi)

        samples_r[k, 5] = phi
        if theta < 0 or theta > np.pi:
            theta = np.pi - (theta % np.pi)
        samples_r[k, 4] = theta

        # compute new orbit after kick
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
            a=poskiorb.utils.P_to_a(samples_r[k, 0], samples_r[k, 1], samples_r[k, 2]),
            m1=samples_r[k, 1],
            m2=samples_r[k, 2],
            m1_remnant_mass=kwargs["M_BH"],
            w=samples_r[k, 3],
            theta=samples_r[k, 4],
            phi=samples_r[k, 5],
            ids=np.ones(1),
        )

        # compute likelihood
        args = [
            samples_r[k, 0],
            samples_r[k, 1],
            samples_r[k, 2],
            samples_r[k, 3],
            samples_r[k, 4],
            samples_r[k, 5],
        ]
        ll = log_likelihood(args, **kwargs)

        if not np.isfinite(ll):
            continue

        samples1.append(
            [float(samples_r[k,0]), float(poskiorb.utils.P_to_a(samples_r[k, 0], samples_r[k, 1], samples_r[k, 2])),
             float(samples_r[k,1]), float(samples_r[k,2]), float(samples_r[k,3]), float(samples_r[k,4]), float(samples_r[k,5])]
        )
        samples2.append(
            [float(p_post), float(e), np.rad2deg(np.arccos(float(cos_i))), float(v_sys), ll]
        )
    print("done !")

    # use numpy arrays
    samples_pre = np.asarray(samples1)
    samples_post = np.asarray(samples2)
    print("sample shapes pre, post-CC:", samples_pre.shape, samples_post.shape)

    with h5py.File(output_filename, "w") as f:
        f.create_dataset("mcmc/pre-cc", data=samples_pre, compression="gzip")
        f.create_dataset("mcmc/post-cc", data=samples_post, compression="gzip")


if __name__ == "__main__":
    # parse command line arguments
    args = parse_args()

    main(config_file=args.config_file)
