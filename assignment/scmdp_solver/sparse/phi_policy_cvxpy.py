from cvxpy import *
#import cvxpy as cvx
import numpy as np

def policy(G, R, L, d, U_next, gamma):
    [n,A]=R.shape
    [m,temp]=d.shape

    Q = Variable(n,A)
    M = Variable(n,n)
    S = Variable(m,n)
    K = Variable(m,m)
    U = Variable(n,1)
    y = Variable(m,1)
    r = Variable(n,1)
    xi = Variable(m,1)
    z = Variable(1,1)

    Kr = kron(np.ones((A,1)),np.eye(n))


    # Create two constraints.
    constraints = [-M + mul_elemwise(G,(np.ones((n, 1)) * vec(Q).T)) * Kr == 0,
                   -r + mul_elemwise(R,Q) * np.ones((A, 1)) == 0,
                   -L.T * y + z * np.ones((n, 1)) - U <= 0,
                   -U + r + gamma * M.T * U_next == 0,
                   -K * L + L * M + S + xi * np.ones((1, n)) == 0,
                   xi + d - K * d >= 0,
                   Q * np.ones((A, 1)) - np.ones((n, 1)) == 0,
                   Q >= np.zeros((n, A)),
                   y >= np.zeros((m, 1)),
                   S >= np.zeros((m, n)),
                   K >= np.zeros((m, m)),
                   ]

    # Form objective.
    obj = Minimize((d.T)*y-z)

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    return U.value, Q.value, M.value, ((d.T) * y - z).value
