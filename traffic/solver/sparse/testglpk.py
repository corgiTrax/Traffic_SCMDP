import numpy as np
from cvxopt import spmatrix, matrix, solvers, glpk
from scipy import sparse
import cvxtool


c = sparse.lil_matrix(np.ones((3,1)))
c = matrix(c.toarray())

Aineq = sparse.lil_matrix(-1 * np.eye(3))
Aineq = cvxtool.scipy_sparse_to_spmatrix(Aineq.tocoo())

bineq = np.zeros((3,1))
bineq[0,0] = 1
bineq[1,0] = 2
bineq[2,0] = 3
bineq = matrix(bineq)

Aeq = sparse.lil_matrix(np.ones((1,3)))
Aeq = cvxtool.scipy_sparse_to_spmatrix(Aeq.tocoo())

beq = np.zeros((1,1))
beq[0,0] = -6
beq = matrix(beq)

sol = solvers.lp(c, Aineq, bineq, Aeq, beq, solver = 'glpk')
print(sol['x'])
print(sol['primal objective'])
