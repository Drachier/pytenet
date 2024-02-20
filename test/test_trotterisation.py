import unittest
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
        """
        Apply the trotter step to the sites 0 and 1, which are the two leftmost sites.
        """
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
        """
        Apply the trotter step to the sites 2 and 3, which are two sites not at the boundary of the MPS.
        """
        acting_on = (2,3)
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on,
                                      self.time_step_size,
                                      direction="sqrt")
        full_operator = np.kron(np.eye(4),trotterstep.exponential_operator)
        full_operator = np.kron(full_operator,np.eye(2))
        expected_resulting_state = full_operator @ self.vectorized
        found_resultig_state = trotterstep.apply_to_mps(self.psi).as_vector()
        self.assertTrue(np.allclose(expected_resulting_state,
                                    found_resultig_state))

    def test_apply_to_mps_2_rightmost_sites(self):
        """
        Apply the trotter step to the sites 3 and 4, which are the two rightmost sites.
        """
        acting_on = (3,4)
        trotterstep = ptn.TrotterStep(self.random_operator,
                                      acting_on,
                                      self.time_step_size,
                                      direction="left")
        full_operator = np.kron(np.eye(8),trotterstep.exponential_operator)
        expected_resulting_state = full_operator @ self.vectorized
        found_resultig_state = trotterstep.apply_to_mps(self.psi).as_vector()
        self.assertTrue(np.allclose(expected_resulting_state,
                                    found_resultig_state))
        
class TestTrotterisation(unittest.TestCase):

    def setUp(self) -> None:
        self.time_step_size = 0.02
        self.num_sites = 5

    def test_brickwall_only_two_site_interaction(self):
        random_twosite_operator = ptn.crandn((2,2,2,2))
        zero_single_site_operator = np.zeros((2,2))
        brickwall_circuit = ptn.Trotterisation.shift_invariant_hamiltonian_brickwall(random_twosite_operator,
                                                                                     zero_single_site_operator,
                                                                                     self.time_step_size,
                                                                                     self.num_sites)
        random_twosite_operator = np.reshape(random_twosite_operator,
                                             (4,4))
        exponentiated_twosite = expm(-1j*self.time_step_size*random_twosite_operator)
        site_list = [(0,1),(2,3),(1,2),(3,4)]
        print([trotter.acting_on for trotter in brickwall_circuit])
        self.assertEqual(4,len(brickwall_circuit))
        for i, sites in enumerate(site_list):
            trotter_step = brickwall_circuit[i]
            self.assertEqual(sites, trotter_step.acting_on)
            self.assertTrue(np.allclose(exponentiated_twosite,trotter_step.exponential_operator))

if __name__ == '__main__':
    unittest.main()
