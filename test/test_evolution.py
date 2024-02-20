import unittest
import copy
import numpy as np
from scipy.linalg import expm
import pytenet as ptn


class TestEvolution(unittest.TestCase):

    def test_tdvp_approximation(self):
        rng = np.random.default_rng()

        # number of lattice sites
        L = 10

        # time step can have both real and imaginary parts;
        # for real-time evolution use purely imaginary dt!
        dt = 0.02 - 0.05j
        # number of steps
        numsteps = 12

        # construct matrix product operator representation of Heisenberg Hamiltonian
        J =  4.0/3
        D =  5.0/13
        h = -2.0/7
        mpoH = ptn.heisenberg_xxz_mpo(L, J, D, h)

        # fix total spin quantum number of wavefunction (trailing virtual bond)
        spin_tot = 2
        # enumerate all possible virtual bond quantum numbers (including multiplicities);
        # will be implicitly reduced by orthonormalization steps below
        qD = [np.array([0])]
        for i in range(L - 1):
            qD.append(np.sort(np.array([q + mpoH.qd[i] for q in qD[-1]]).reshape(-1)))
        qD.append(np.array([2*spin_tot]))

        # initial wavefunction as MPS with random entries
        psi = ptn.MPS(mpoH.qd, qD, fill='random', rng=rng)
        psi.orthonormalize(mode='left')
        psi.orthonormalize(mode='right')
        # effectively clamp virtual bond dimension of initial state
        Dinit = 8
        for i in range(L):
            psi.A[i][:, Dinit:, :] = 0
            psi.A[i][:, :, Dinit:] = 0
        # orthonormalize again
        psi.orthonormalize(mode='left')
        self.assertEqual(psi.qD[-1][0], 2*spin_tot,
            msg='trailing bond quantum number must not change during orthonormalization')

        # total spin operator as MPO
        Szgraph = ptn.OpGraph.from_opchains(
            [ptn.OpChain([1], [0, 0], 1.0, istart) for istart in range(L)], L, 0)
        Sztot = ptn.MPO.from_opgraph(mpoH.qd[0], Szgraph, { 0: np.identity(2), 1: np.diag([0.5, -0.5]) })

        # explicitly compute average spin
        spin_avr = ptn.operator_average(psi, Sztot)
        self.assertAlmostEqual(spin_avr, spin_tot, delta=1e-14,
            msg='average spin must be equal to prescribed value')

        # reference time evolution
        psi_ref = expm(-dt*numsteps * mpoH.as_matrix()) @ psi.as_vector()

        # run TDVP time evolution
        psi1 = copy.deepcopy(psi)
        psi2 = copy.deepcopy(psi)
        ptn.integrate_local_singlesite(mpoH, psi1, dt, numsteps, numiter_lanczos=5)
        ptn.integrate_local_twosite(mpoH, psi2, dt, numsteps, numiter_lanczos=10)

        # compare time-evolved wavefunctions
        self.assertTrue(np.allclose(psi1.as_vector(), psi_ref, atol=2e-5),
            msg='time-evolved wavefunction obtained by single-site TDVP time evolution must match reference')
        self.assertTrue(np.allclose(psi2.as_vector(), psi_ref, atol=1e-10),
            msg='time-evolved wavefunction obtained by two-site TDVP time evolution must match reference')


    def test_tdvp_symmetry(self):
        rng = np.random.default_rng()

        # number of lattice sites
        L = 10

        # real-time evolution
        dt = 0.5j

        # construct matrix product operator representation of Heisenberg Hamiltonian
        J =  4.0/3
        D =  5.0/13
        h = -2.0/7
        mpoH = ptn.heisenberg_xxz_mpo(L, J, D, h)
        mpoH.zero_qnumbers()

        # quantum numbers not used here; set them to zero
        qD = [np.array([0])]
        for _ in range(L - 1):
            qD.append(np.zeros(5, dtype=int))
        qD.append(np.array([0]))

        # initial wavefunction as MPS with random entries
        psi = ptn.MPS(mpoH.qd, qD, fill='random', rng=rng)
        psi.orthonormalize(mode='left')

        psi_ref = psi.as_vector()

        # evolve forward and then backward in time;
        # should arrive at initial state since integration method is symmetric
        psi1 = copy.deepcopy(psi)
        ptn.integrate_local_singlesite(mpoH, psi1,  dt, 1, numiter_lanczos=10)
        ptn.integrate_local_singlesite(mpoH, psi1, -dt, 1, numiter_lanczos=10)
        psi2 = copy.deepcopy(psi)
        ptn.integrate_local_twosite(mpoH, psi2,  dt, 1, numiter_lanczos=10, tol_split=1e-10)
        ptn.integrate_local_twosite(mpoH, psi2, -dt, 1, numiter_lanczos=10, tol_split=1e-10)

        # compare
        self.assertTrue(np.allclose(psi1.as_vector(), psi_ref, atol=1e-10))
        # larger deviation for two-site TDVP presumably due to varying bond dimensions
        self.assertTrue(np.allclose(psi2.as_vector(), psi_ref, atol=1e-6))

    def test_tebd_approximation(self):
        rng = np.random.default_rng()

        # number of lattice sites
        L = 5

        # real-time evolution
        dt = 0.01
        # number of steps
        numsteps = 12

        # Construct Trotterised Brickwall Circuit
        J =  4.0/3
        h = -2.0/7
        g = 0.5
        paulix = np.array([[0, 1], [1, 0]], dtype=complex)
        pauliz = np.array([[1, 0], [0, -1]], dtype=complex)
        two_site_operator = np.kron(pauliz,pauliz)
        single_site_operator = h * pauliz + g * paulix
        trotterisation = ptn.Trotterisation.shift_invariant_hamiltonian_brickwall(two_site_operator,
                                                                                  single_site_operator,
                                                                                  dt,
                                                                                  L)
        for trotterstep in trotterisation:
            trotterstep.direction = 'right'

        # initial wavefunction as MPS with random entries
        qd = [[0,0] for _ in range(L)]
        qD = [[0]]
        bond_dim = 20
        qD.extend([bond_dim*[0] for _ in range(L-1)])
        qD.append([0])
        psi = ptn.MPS(qd, qD, fill='random', rng=rng)
        psi.orthonormalize(mode='left')
        psi.orthonormalize(mode='right')

        # Reference Computation
        H = ptn.ising_mpo(L, J, h, g).as_matrix()
        psi_ref = expm(-1j * dt * numsteps * H) @ psi.as_vector()

        # run TEBD time evolution
        psi1 = copy.deepcopy(psi)
        ptn.tebd_evolution(trotterisation, psi1, numsteps, tol_split=0)

        # compare time-evolved wavefunctions
        print(max(np.abs(psi1.as_vector() - psi_ref)))
        self.assertTrue(np.allclose(psi1.as_vector(), psi_ref, atol=1e-10),
            msg='time-evolved wavefunction obtained by TEBD time evolution must match reference')



if __name__ == '__main__':
    unittest.main()
