import unittest
import copy
import numpy as np
from scipy.linalg import expm
import pytenet as ptn

class TestTrotterStepInit(unittest.TestCase):
    def setUp(self) -> None:
        self.time_step_size = 0.02
  
    def test_exponentiation(self):
        random_operator = ptn.crandn((2,2,2,2))
        trotterstep = ptn.TrotterStep(random_operator,
                                      acting_on=(0,1),
                                      time_step_size=self.time_step_size)
        found_exponential = trotterstep.exponential_operator
        reshaped_operator = random_operator.reshape(4,4)
        expected_exponential = expm(-1j*self.time_step_size*reshaped_operator)
        self.assertTrue(np.allclose(found_exponential, expected_exponential))

class TestTrotterStepMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.time_step_size = 0.02
        qd = 2*[0]
        num_sites = 5
        qD = [4*[0] for _ in range(num_sites)]
        self.psi = ptn.MPS(qd, qD, fill='random')
        self.psi.orthonormalize(mode='left')
        


if __name__ == '__main__':
    unittest.main()