
from typing import Sequence

from timeit import default_timer as timer
from numpy import abs
from matplotlib.pyplot import subplots, show
from tqdm import tqdm

from pytenet.mps import MPS
from pytenet.operation import vdot
from pytenet.compression import rounding

from create_mps import get_high_bd_ising_state

def mps_error(mps1: MPS, mps2: MPS) -> float:
    """
    Computes the error between two MPS.
    """
    return abs(1-vdot(mps1, mps2))

def run_compression_for_tolerance_range(nsites: int, tol_range: Sequence[float],
                                        compress_algorithm: callable, **kwargs) -> dict:
    """
    Runs the compression algorithm for a range of tolerances.
    """
    mps = get_high_bd_ising_state(nsites)
    print("High bond dimension state generated.")
    print("Starting compression algorithm.")
    results = {"tolerance": [], "error": [], "compression_time": []}
    for tol in tol_range:
        start = timer()
        compr_mps = compress_algorithm(mps, tol, **kwargs)
        end = timer()
        results["tolerance"].append(tol)
        error = mps_error(mps, compr_mps)
        results["error"].append(error)
        results["compression_time"].append(end-start)
    print("Compression algorithm finished.")
    return results

def run_compression_for_site_range(site_range: Sequence[int], tol_range: Sequence[float],
                                    compress_algorithm: callable, **kwargs) -> dict:
    """
    Runs the compression algorithm for a range of site numbers.
    """
    results = {"site_number": [], "tolerance": [], "error": [], "compression_time": []}
    for nsites in tqdm(site_range):
        result = run_compression_for_tolerance_range(nsites,
                                                     tol_range,
                                                     compress_algorithm,
                                                     **kwargs)
        results["site_number"].append(nsites)
        results["tolerance"] = result["tolerance"]
        results["error"].append(result["error"])
        results["compression_time"].append(result["compression_time"])
    return results

def run_compression(compress_algorithm: callable, **kwargs) -> dict:
    """
    Runs the compression algorithm for a range of site numbers and tolerances.
    """
    site_range = [10, 20]
    tol_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    return run_compression_for_site_range(site_range,
                                          tol_range,
                                          compress_algorithm,
                                          **kwargs)

def plotting(results: dict):
    """
    Plots the results of the compression algorithm.
    """
    fig, ax = subplots(1, 1)
    for i, nsites in enumerate(results["site_number"]):
        ax.plot(results["tolerance"], results["error"][i], label=f"{nsites} sites")
    ax.set_xlabel("Tolerance $\varepsilon$")
    ax.set_ylabel("Error")
    ax.set_xscale("log")

    show()

if __name__ == "__main__":
    results = run_compression(compress_algorithm=rounding)
    plotting(results)