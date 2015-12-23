import cvxpy
import numpy as np

def policy(G, R, L, d, x, U_next, U_ref, opt_ref, gamma):
    opt_ref += 10e-4
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
                   -K * L + L * M  + xi * np.ones((1, n)) <= 0,
                   xi + d - K * d >= 0,
                   Q * np.ones((A, 1)) - np.ones((n, 1)) == 0,
                   Q >= 0, #np.zeros((n, A)),
                   y >= 0, # np.zeros((m, 1)),
                   K >= 0, #np.zeros((m, m)),
                   ]

    # Form objective.
    obj = cvxpy.Minimize(-(x.T) * U)

    # Form and solve problem.
    prob = cvxpy.Problem(obj, constraints)
#    prob.solve(solver = cvxpy.MOSEK, verbose = True)
    prob.solve(solver = cvxpy.ECOS, verbose = False, max_iters = 500)
#    prob.solve(solver = cvxpy.SCS, verbose = False) 

    return U.value, Q.value, M.value

def policy_unU(G, R, L, d, x_prev, un_U, gamma):
    [n,A]=R.shape
    [m,temp]=d.shape

    Q = cvxpy.Variable(n,A)
    M = cvxpy.Variable(n,n)
    S = cvxpy.Variable(m,n)
    K = cvxpy.Variable(m,m)
#    U = cvxpy.Variable(n,1)
#    y = cvxpy.Variable(m,1)
#    r = cvxpy.Variable(n,1)
    xi = cvxpy.Variable(m,1)
#    z = cvxpy.Variable(1,1)
    x = cvxpy.Variable(n,1)

    Kr = cvxpy.kron(np.ones((A,1)),np.eye(n))

    # Create two constraints.
    constraints = [#(d.T) * y - z <= opt_ref,
             -M + cvxpy.mul_elemwise(G,(np.ones((n, 1)) * cvxpy.vec(Q).T)) * Kr == 0,
#                   -r + cvxpy.mul_elemwise(R,Q) * np.ones((A, 1)) == 0,
#                   -L.T * y + z * np.ones((n, 1)) - U_ref <= 0,
#                   -U + r + gamma * M.T * U_next == 0,
                   -K * L + L * M  + xi * np.ones((1, n)) <= 0,
                   xi + d - K * d >= 0,
                   x - M * x_prev == 0,
                   Q * np.ones((A, 1)) - np.ones((n, 1)) == 0,
                   Q >= 0, #np.zeros((n, A)),
 #                  y >= 0, # np.zeros((m, 1)),
                   K >= 0, #np.zeros((m, m)),
                   x >= 0,
                   x.T * np.ones((n, 1)) - 1 == 0
                   ]

    # Form objective.
    obj = cvxpy.Minimize(-(x.T) * un_U)

    # Form and solve problem.
    prob = cvxpy.Problem(obj, constraints)
#    prob.solve(solver = cvxpy.MOSEK, verbose = True)
    prob.solve(solver = cvxpy.ECOS, verbose = False,  max_iters = 500)
#    prob.solve(solver = cvxpy.SCS, verbose = False) 

    return Q.value, M.value, x.value

def policy_unU2(G, R, L, d, x_prev, un_U, gamma):
    # a greedy policy that max next step return using un_U 
    [n,A]=R.shape
    [m,temp]=d.shape

    Q = cvxpy.Variable(n,A)
    M = cvxpy.Variable(n,n)
    x = cvxpy.Variable(n,1)

    Kr = cvxpy.kron(np.ones((A,1)),np.eye(n))

    # Create two constraints.
    constraints = [-M + cvxpy.mul_elemwise(G,(np.ones((n, 1)) * cvxpy.vec(Q).T)) * Kr == 0,
                   L * x -  d <= 0,
                   x - M * x_prev == 0,
                   Q * np.ones((A, 1)) - np.ones((n, 1)) == 0,
                   Q >= 0, 
                   x >= 0,
                   x.T * np.ones((n, 1)) - 1 == 0
                   ]

    # Form objective.
    obj = cvxpy.Minimize(-(x.T) * un_U)

    # Form and solve problem.
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver = cvxpy.ECOS, verbose = False,  max_iters = 500)
#    prob.solve(solver = cvxpy.SCS, verbose = False) 

    return Q.value, M.value, x.value
