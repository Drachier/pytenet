import unittest
import numpy as np
import sys
sys.path.append('../pytenet/')
from mps import MPS, merge_MPS_tensor_pair, split_MPS_tensor
from qnumber import qnumber_outer_sum, qnumber_flatten, is_qsparse


class TestMPS(unittest.TestCase):

    def test_orthonormalization(self):

        # create random matrix product state
        d = 7
        D = [1, 4, 15, 13, 7, 1]
        mps0 = MPS(np.random.randint(-2, 3, size=d), [np.random.randint(-2, 3, size=Di) for Di in D], fill='random')

        self.assertEqual(mps0.bond_dims, D, msg='virtual bond dimensions')

        # wavefunction on full Hilbert space
        psi = mps0.as_vector()

        # performing left-orthonormalization...
        cL = mps0.orthonormalize(mode='left')

        self.assertLessEqual(mps0.bond_dims[1], d,
            msg='virtual bond dimension can only increase by a factor of "d" per site')

        for i in range(mps0.nsites):
            self.assertTrue(is_qsparse(mps0.A[i], [mps0.qd, mps0.qD[i], -mps0.qD[i+1]]),
                            msg='sparsity pattern of MPS tensors must match quantum numbers')

        psiL = mps0.as_vector()
        # wavefunction should now be normalized
        self.assertAlmostEqual(np.linalg.norm(psiL), 1., delta=1e-12, msg='wavefunction normalization')

        # wavefunctions before and after left-normalization must match
        # (up to normalization factor)
        self.assertAlmostEqual(np.linalg.norm(cL*psiL - psi), 0., delta=1e-10,
                               msg='wavefunctions before and after left-normalization must match')

        # check left-orthonormalization
        for i in range(mps0.nsites):
            s = mps0.A[i].shape
            assert s[0] == d
            Q = mps0.A[i].reshape((s[0]*s[1], s[2]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[2])), 0., delta=1e-12,
                                   msg='left-orthonormalization')

        # performing right-orthonormalization...
        cR = mps0.orthonormalize(mode='right')

        self.assertLessEqual(mps0.bond_dims[-2], d,
            msg='virtual bond dimension can only increase by a factor of "d" per site')

        for i in range(mps0.nsites):
            self.assertTrue(is_qsparse(mps0.A[i], [mps0.qd, mps0.qD[i], -mps0.qD[i+1]]),
                            msg='sparsity pattern of MPS tensors must match quantum numbers')

        self.assertAlmostEqual(abs(cR), 1., delta=1e-12,
            msg='normalization factor must have magnitude 1 due to previous left-orthonormalization')

        psiR = mps0.as_vector()
        # wavefunctions must match
        self.assertAlmostEqual(np.linalg.norm(psiL - cR*psiR), 0., delta=1e-10,
                               msg='wavefunctions after left- and right-orthonormalization must match')

        # check right-orthonormalization
        for i in range(mps0.nsites):
            s = mps0.A[i].shape
            assert s[0] == d
            Q = mps0.A[i].transpose((0, 2, 1)).reshape((s[0]*s[2], s[1]))
            QH_Q = np.dot(Q.conj().T, Q)
            self.assertAlmostEqual(np.linalg.norm(QH_Q - np.identity(s[1])), 0., delta=1e-12,
                                   msg='right-orthonormalization')


    def test_split_tensor(self):

        # physical dimensions
        d0, d1 = 3, 5
        # outer virtual bond dimensions
        D0, D2 = 14, 17

        Apair = randn_complex((d0*d1, D0, D2)) / np.sqrt(d0*d1*D0*D2)

        # fictitious quantum numbers
        qd0 = np.random.randint(-2, 3, size=d0)
        qd1 = np.random.randint(-2, 3, size=d1)
        qD = [np.random.randint(-2, 3, size=D0), np.random.randint(-2, 3, size=D2)]

        # enforce block sparsity structure dictated by quantum numbers
        mask = qnumber_outer_sum([qnumber_flatten([qd0, qd1]), qD[0], -qD[1]])
        Apair = np.where(mask == 0, Apair, 0)

        for svd_distr in ['left', 'right', 'sqrt']:
            (A0, A1, qbond) = split_MPS_tensor(Apair, qd0, qd1, qD, svd_distr=svd_distr, tol=0)

            self.assertTrue(is_qsparse(A0, [qd0, qD[0], -qbond]),
                            msg='sparsity pattern of A0 tensors must match quantum numbers')
            self.assertTrue(is_qsparse(A1, [qd1, qbond, -qD[1]]),
                            msg='sparsity pattern of A1 tensors must match quantum numbers')

            # merged tensor must agree with the original tensor
            Amrg = merge_MPS_tensor_pair(A0, A1)
            self.assertAlmostEqual(np.linalg.norm(Amrg - Apair), 0., delta=1e-13,
                                   msg='splitting and subsequent merging must give the same tensor')


def randn_complex(size):
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)


if __name__ == '__main__':
    unittest.main()
