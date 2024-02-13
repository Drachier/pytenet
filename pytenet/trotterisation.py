from __future__ import annotations
from typing import Tuple, List, Union
from numbers import Number

import numpy as np

class BlockDecimationStep():

    def __init__(self, operator: np.ndarray,
                 acting_on: Union[int, Tuple[int, ]],
                 factor: Number = 1,
                 swaps_before: Union[None,Tuple[int,int],List[Tuple[int,int]]]=None,
                 swaps_after: Union[None,Tuple[int,int],List[Tuple[int,int]]]=None):

        self.operator = operator
        if isinstance(acting_on, int):
            acting_on = (acting_on,)
        self.acting_on = acting_on
        assert len(acting_on) == operator.ndim // 2
        self.swaps_before = self._init_swap_list(swaps_before)
        self.swaps_after = self._init_swap_list(swaps_after)
        self.factor = factor

    def _init_swap_list(self,
                        swaps: Union[None,Tuple[int,int],List[Tuple[int,int]]]) -> List[int,int]:
        if swaps is None:
            return []
        if isinstance(swaps, tuple):
            swaps = [swaps]
        return swaps

class BlockDecimation():

    def __init__(self, hamiltonian_terms, swaps)