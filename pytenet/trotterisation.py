from __future__ import annotations
from typing import Tuple, List, Union, Self
from numbers import Number
from warnings import warn

import numpy as np
from scipy.linalg import expm

from .mps import MPS, merge_mps_tensor_pair, split_mps_tensor

class TrotterStep():
    """
    A single step of a Trotterisation of a Hamiltonian. It is defined
     by the following attributes:
    
    Args:
        operator (np.ndarray): The operator to be applied in this step.
        It is the exponent in the Trotter formula.
            exp(-i * factor* dt * H) 
        The dimensions of the operator have to be even. The first half
        of the dimensions are the output dimensions, the second half
        are the input dimensions. The legs should by in the order of
        the sites they are applied to.
        acting_on (Union[int, Tuple[int,int]]): The indices of the
        first and last site+1 to be acted upon by the operator. The
        syntax is the same as for slicing in Python. If an integer
        is given, the operator is applied to the site with this index.
        time_step_size (Number): The size of the time step dt.
        factor (Number): A factor by which the operator is multiplied.
        swaps_before (Union[None,Tuple[int,int],List[Tuple[int,int]]]):
        A list of sites that should be swapped before the exponent is
        applied to the state. The sites must be neighbours.
        swaps_after (Union[None,Tuple[int,int],List[Tuple[int,int]]]):
        A list of sites that should be swapped after the exponent is
        applied to the state. The sites must be neighbours.
        direction (Union[None,str]): The direction in which the
        singular values are distributed in the SVD. Relevant to keep
        canonical forms intact. If None, the direction has to be
        specified during application of the Trotter step.
        operator_exponentiated (bool): If True, the operator is already
        exponentiated. If False, the operator is exponentiated during
        initialization of the Trotter step. Default is False.

           0|  1|      2|
         ___|___|_______|_______
        |                       |
        |        operator       |
        |_______________________|
            |   |       |
           4|  5|      6|
         siten siten+1 siten+2

        In this case acting on would be (n,n+3).
    """

    def __init__(self, operator: np.ndarray,
                 acting_on: Union[int, Tuple[int,int]],
                 time_step_size: Number,
                 factor: Number = 1,
                 swaps_before: Union[None,Tuple[int,int],List[Tuple[int,int]]]=None,
                 swaps_after: Union[None,Tuple[int,int],List[Tuple[int,int]]]=None,
                 direction: Union[None,str] = None,
                 operator_exponentiated: bool = False):
        """
        Initialize the Trotter step.
        """
        if isinstance(acting_on, int):
            acting_on = (acting_on,acting_on+1)
        self.acting_on = acting_on
        if operator_exponentiated:
            assert operator.ndim == 2
        else:
            assert len(acting_on) == operator.ndim // 2
        self._time_step_size = time_step_size
        self.swaps_before = self._init_swap_list(swaps_before)
        self.swaps_after = self._init_swap_list(swaps_after)
        self._factor = factor
        if operator_exponentiated:
            self.exponential_operator = operator
        else:
            self.exponential_operator = self._exponentiate_operator(operator)
        self.direction = direction

    @property
    def time_step_size(self) -> Number:
        """
        The size of the time step dt.
        """
        return self._time_step_size

    @time_step_size.setter
    def time_step_size(self, value: Number) -> None:
        """
        Set the size of the time step dt.

        This means the exponential operator has to be
        updated.
        """
        self._time_step_size = value
        self.exponential_operator = self._exponentiate_operator(self.exponential_operator)

    @property
    def factor(self) -> Number:
        """
        A factor by which the operator is multiplied.
        """
        return self._factor

    @factor.setter
    def factor(self, value: Number) -> None:
        """
        Set the factor by which the operator is multiplied.

        This means the exponential operator has to be
        updated.
        """
        self._factor = value
        self.exponential_operator = self._exponentiate_operator(self.exponential_operator)

    def _init_swap_list(self,
                        swaps: Union[None,Tuple[int,int],List[Tuple[int,int]]]) -> List[Tuple[int,int]]:
        """
        Unifies different possible SWAP inputs.
        """
        if swaps is None:
            return []
        warn("SWAPs are not yet used in the implementation!")
        if isinstance(swaps, tuple):
            swaps = [swaps]
        for swap in swaps:
            assert len(swap) == 2, "Each swap must be a tuple of length 2!"
            assert abs(swap[0] - swap[1]) == 1, "The sites to be swapped must be neighbours!"
        return swaps

    def _exponentiate_operator(self,
                               operator: np.ndarray) -> np.ndarray:
        """
        Exponentiate the operator.
        """
        factor = -1j * self.factor * self.time_step_size
        ndim_half = operator.ndim // 2
        matrix = np.reshape(operator, (ndim_half, ndim_half))
        return expm(factor * matrix)

    def apply_to_mps(self,
                     mps: MPS,
                     tol: float,
                     svd_distr: Union[None, str] = None) -> MPS:
        """
        Apply the Trotter step to an MPS.

        Args:
            mps (MPS): The MPS to which the Trotter step is to be applied.
            tol (float): The tolerance for the SVD.
            svd_distr (str): The distribution of the singular values
            in the SVD.
        
        Returns:
            MPS: The MPS after the application of the Trotter step.
            This is not a copy but the same object.
        """
        if svd_distr is None:
            if self.direction is None:
                errstr = "The direction of the SVD has to be specified!"
                raise ValueError(errstr)
            svd_distr = self.direction
        if self.exponential_operator.ndim == 2:
            A = mps.A[self.acting_on[0]]
            A = np.tensordot(self.exponential_operator, A, axes=(1,0))
        elif self.exponential_operator.ndim == 4:
            leftsite = self.acting_on[0]
            rightsite = self.acting_on[1]
            Amerged = merge_mps_tensor_pair(mps.A[leftsite],
                                            mps.A[rightsite])
            Amerged = np.tensordot(self.exponential_operator,
                                   Amerged,
                                   axes=(1,0))
            A = split_mps_tensor(Amerged, mps.qd, mps.qd,
                                 [mps.qD[leftsite], mps.qD[rightsite+1]],
                                 svd_distr, tol)
            mps.A[leftsite], mps.A[rightsite], mps.qD[rightsite] = A
        else:
            errstr = "Only operators of dimension 2 or 4 are supported!"
            raise NotImplementedError(errstr)
        return mps

class Trotterisation(list):
    """
    This class can be considered a factory class to create Trotter step lists
    for given Hamiltonians.
    """

    def __init__(self, trotter_steps: List[TrotterStep]):
        """
        A list of Trotter steps that are to be performed in sequence.

        Args:
            trotter_steps (List[TrotterStep]): A list of Trotter steps
            that are to be performed in sequence.
        """
        super().__init__(trotter_steps)

    @classmethod
    def shift_invariant_hamiltonian_brickwall(cls,
                                              two_site_operator: np.ndarray,
                                              single_site_operator: np.ndarray,
                                              time_step_size: Number,
                                              num_sites: int) -> Self:
        """
        Create a Trotterisation for a shift-invariant Hamiltonian.
        """
        trotter_steps = []
        extended_single_site_operator = np.kron(single_site_operator,
                                                np.eye(single_site_operator.shape[0]))
        total_operator = two_site_operator + extended_single_site_operator
        exp_operator = expm(-1j * time_step_size * total_operator)
        # Even Sites
        for i in range(num_sites - 1,2):
            trotter_step = TrotterStep(exp_operator,
                                       (i,i+1),
                                       time_step_size,
                                       operator_exponentiated=True)
            trotter_steps.append(trotter_step)
        # Odd Sites
        for i in range(1,num_sites-1,2):
            trotter_step = TrotterStep(exp_operator,
                                       (i,i+1),
                                       time_step_size,
                                       operator_exponentiated=True)
            trotter_steps.append(trotter_step)
        # Final Single Site Operator
        final_exp_operator = expm(-1j * time_step_size * single_site_operator)
        trotter_step = TrotterStep(final_exp_operator,
                                   num_sites-1,
                                   time_step_size,
                                   operator_exponentiated=True)
        trotter_steps.append(trotter_step)
        return cls(trotter_steps)

