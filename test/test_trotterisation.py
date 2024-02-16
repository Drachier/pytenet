import unittest
import copy
import numpy as np
from scipy.linalg import expm
import pytenet as ptn

class TestTrotterStepInit(unittest.TestCase):
    def setUp(self) -> None:
        self.time_step_size = 0.02
        self.random_operator = ptn.crandn((2,2,2,2))

    def test_exponentiation(self):
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on=(0,1),
                                      time_step_size=self.time_step_size)
        found_exponential = trotterstep.exponential_operator
        reshaped_operator = self.random_operator.reshape(4,4)
        expected_exponential = expm(-1j*self.time_step_size*reshaped_operator)
        self.assertTrue(np.allclose(expected_exponential, found_exponential))

class TestTrotterStepMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.time_step_size = 0.02
        qd = 2*[0]
        num_sites = 5
        qD = [[0]]
        qD.extend([4*[0] for _ in range(num_sites-1)])
        qD.append([0])
        self.psi = ptn.MPS(qd, qD, fill='random')
        self.psi.orthonormalize(mode='left')
        self.vectorized = self.psi.as_vector()
        self.random_operator = ptn.crandn((2,2,2,2))

    def test_apply_to_mps_2_leftmost_sites(self):
        acting_on = (0,1)
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on,
                                      self.time_step_size,
                                      direction="right")
        full_operator = np.kron(trotterstep.exponential_operator,np.eye(8))
        expected_resulting_state = full_operator @ self.vectorized
        found_resultig_state = trotterstep.apply_to_mps(self.psi).as_vector()
        self.assertTrue(np.allclose(expected_resulting_state,
                                    found_resultig_state))

    def test_apply_to_mps_2_middle_sites(self):
        acting_on = (2,3)
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on,
                                      self.time_step_size,
                                      direction="right")
        full_operator = np.kron(np.eye(4),trotterstep.exponential_operator)
        full_operator = np.kron(full_operator,np.eye(2))
        expected_resulting_state = full_operator @ self.vectorized
        found_resultig_state = trotterstep.apply_to_mps(self.psi).as_vector()
        self.assertTrue(np.allclose(expected_resulting_state,
                                    found_resultig_state))

    def test_apply_to_mps_2_rightmost_sites(self):
        acting_on = (3,4)
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on,
                                      self.time_step_size,
                                      direction="right")
        full_operator = np.kron(np.eye(8),trotterstep.exponential_operator)
        expected_resulting_state = full_operator @ self.vectorized
        found_resultig_state = trotterstep.apply_to_mps(self.psi).as_vector()
        self.assertTrue(np.allclose(expected_resulting_state,
                                    found_resultig_state))


if __name__ == '__main__':
    unittest.main()
