import cvxpy
#import cvxpy as cvx
import numpy as np

def policy(G, R, L, d, un_Q, un_M, un_U, U_next, U_ref, opt_ref, gamma):
    [n,A]=R.shape
    [m,temp]=d.shape

    Q = cvxpy.Variable(n,A)
    M = cvxpy.Variable(n,n)
    S = cvxpy.Variable(m,n)
    K = cvxpy.Variable(m,m)
    U = cvxpy.Variable(n,1)
    y = cvxpy.Variable(m,1)
    r = cvxpy.Variable(n,1)
    xi = cvxpy.Variable(m,1)
    z = cvxpy.Variable(1,1)

    Kr = cvxpy.kron(np.ones((A,1)),np.eye(n))


    # Create two constraints.
    constraints = [(d.T) * y - z <= opt_ref,
                   -M + cvxpy.mul_elemwise(G,(np.ones((n, 1)) * cvxpy.vec(Q).T)) * Kr == 0,
                   -r + cvxpy.mul_elemwise(R,Q) * np.ones((A, 1)) == 0,
                   -L.T * y + z * np.ones((n, 1)) - U_ref <= 0,
                   -U + r + gamma * M.T * U_next == 0,
                   -K * L + L * M + S + xi * np.ones((1, n)) == 0,
                   xi + d - K * d >= 0,
                   Q * np.ones((A, 1)) - np.ones((n, 1)) == 0,
                   Q >= np.zeros((n, A)),
                   y >= np.zeros((m, 1)),
                   S >= np.zeros((m, n)),
                   K >= np.zeros((m, m)),
                   M >= np.zeros((n, n))
                   ]

    # Form objective.
    obj = cvxpy.Minimize(cvxpy.norm((Q - un_Q), 'fro'))
#    obj = cvxpy.Minimize(cvxpy.norm((M - un_M), 'fro'))
#    obj = cvxpy.Minimize(cvxpy.norm((U - un_U)))

    # Form and solve problem.
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver = cvxpy.ECOS, verbose = False, max_iters = 1000)

    return U.value, Q.value, M.value
