import operator

import numpy as np

class ExplaiNE(object):
    def __init__(self):
        pass

    def H_diag_block(self, i, P_row_i, X, gamma):
        temp_X = X.copy()
        X[i, :] = 0
        return -gamma**2 * (X.T*(P_row_i*(1-P_row_i))).dot(X)

    def compute_grapd(self, i, j, P_row_i, X, gamma):
        H = self.H_diag_block(i, P_row_i, X, gamma)
        P_ij = P_row_i[j]
        return 1/(P_ij*(1-P_ij))/gamma**2* X[j,:].dot(np.linalg.solve(-H, X.T))

    def explain(self, i, j, A_i, P_row_i, X, gamma):
        grad = self.compute_grapd(i, j, P_row_i, X, gamma)
        nbr_ids = np.setdiff1d(A_i.nonzero()[1], [j])

        return sorted(zip(nbr_ids, grad[nbr_ids]),
              key=operator.itemgetter(1), reverse=True)
