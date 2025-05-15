import numpy as np
from numba import jit
import timeit


def cholesky2diag(cholesky):
    return np.diag(cholesky_vector_to_spd(cholesky))


def stiff2cholesky(diag):
    spd_matrix = np.diag(diag)
    return spd_to_cholesky_vector(spd_matrix)

# @jit("float64[:](float64[:,:])", nopython=True)


def spd_to_cholesky_vector(spd_matrix):
    """
        Compute Cholesky decomposition for SPD matrix.
        Then, extract and return its lower triangle as a vector.
    """
    cholesky_matrix = np.linalg.cholesky(spd_matrix).ravel()
    tril_mask = np.array([0, 3, 4, 6, 7, 8])
    return cholesky_matrix[tril_mask]

# @jit("float64[:,:](float64[:])", nopython=True)


def cholesky_vector_to_spd(cholesky_vector):
    """
        Reconstruct Cholesky decomposition matrix from vector.
        Then compute SPD matrix L * L.T
    """
    cholesky_matrix = np.zeros((3, 3), dtype=np.float64)
    mask = np.tril_indices(cholesky_matrix.shape[0], k=0)
    for i in range(6):
        cholesky_matrix[mask[0][i]][mask[1][i]] = cholesky_vector[i]
    return cholesky_matrix @ cholesky_matrix.T


if __name__ == '__main__':
    from sklearn.datasets import make_spd_matrix
    spd = make_spd_matrix(3)
    print("SPD Matrix: \n", spd, type(spd))
    ch = np.linalg.cholesky(spd)
    print("Cholesky decomposition L:\n", ch)
    print("SPD from L @ L.T\n", np.dot(ch, ch.T))

    print("=== Verification ===")
    ch_v = spd_to_cholesky_vector(spd)
    print("cholesky vector\n", ch_v)
    spd_rec = cholesky_vector_to_spd(ch_v)
    print("reconstructed spd\n", spd_rec)
