
from copy import deepcopy
from numpy import asarray
from argparse import ArgumentParser
from enum import Enum

from pytenet.evolution import integrate_local_twosite
from pytenet.hamiltonian import ising_mpo, long_range_xy_chain_mpo
from pytenet.mps import MPS

def generate_initial_state(nsite: int) -> MPS:
    """
    The initial state will be merely the computational zero state on all sites.

    Args:
        nsite: The number of sites in the MPS.
    """
    phys_dim = 2
    init_bd = 1
    qd = [0 for _ in range(phys_dim)]
    qD = [[0 for _ in range(init_bd)] for _ in range(nsite-1)]
    qD = [[0]] + qD + [[0]]
    mps = MPS(qd, qD, fill='postpone')
    ket_zero = asarray([1, 0], dtype=complex).reshape(2, 1, 1)
    for site in range(nsite):
        mps.A[site] = deepcopy(ket_zero)
    return mps

def get_high_bd_ising_state(nsites: int, tol_split: float) -> MPS:
    """
    Runs the Ising model evolution on the initial state.
    """
    J = 1
    h = 0.2
    g = 0.3
    dt = 0.01
    numsteps = 100
    mpo = ising_mpo(nsites, J, h, g)
    mps = generate_initial_state(nsites)
    integrate_local_twosite(mpo, mps, dt, numsteps, tol_split=tol_split)
    return mps

def get_high_bd_long_range_state(nsites: int, tol_split: float) -> MPS:
    """
    Runs the long-range XY chain evolution on the initial state.
    """
    J = 1
    decay = 0.2
    dt = 0.01
    numsteps = 100
    mpo = long_range_xy_chain_mpo(nsites, J, decay)
    mps = generate_initial_state(nsites)
    integrate_local_twosite(mpo, mps, dt, numsteps, tol_split=tol_split)
    return mps

class Models(Enum):
    ISING = 0
    LONG_RANGE = 1

    @classmethod
    def from_int(cls, model: int):
        if model == 0:
            return cls.ISING
        elif model == 1:
            return cls.LONG_RANGE
        else:
            raise ValueError(f"Unknown model {model}")

    def mpo(self, nsites: int, tol_split: float) -> MPS:
        if self == self.ISING:
            return get_high_bd_ising_state(nsites, tol_split)
        elif self == self.LONG_RANGE:
            return get_high_bd_long_range_state(nsites, tol_split)
        else:
            raise ValueError(f"Unknown model {self}")

    def filename(self, prefix: str = "mps") -> str:
        return f"{prefix}_{self.name.lower()}.npz"

def input_handling():
    parser = ArgumentParser()
    parser.add_argument('--models', nargs='+', type=int, default=[0])
    parser.add_argument('--nsites', type=int, default=10)
    parser.add_argument('--tol_split', type=float, default=1e-10)
    parser.add_argument('--output_file', type=str, default='mps')
    return parser.parse_args()

def main():
    args = input_handling()
    for model in args.models:
        model = Models.from_int(model)
        mps = model.mpo(args.nsites, args.tol_split)
        mps.save(model.filename(args.output_file))

if __name__ == '__main__':
    main()