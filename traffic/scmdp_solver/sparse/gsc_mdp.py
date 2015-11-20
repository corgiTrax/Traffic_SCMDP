# function solves for safe policy: basic version + heuristics(pro, bf)
# Input data:
# transition matrix g(A,n,n)
# reward matrix for each epoch R(T-1,n,A)
# terminal reward RT(n,1)
# upper bound d(m,1)
# generalization matrix L(m,n)
# initial distribution x0(n,1)
# discount factor gamma(1,1)

import copy as cp
import numpy as np
from numpy import linalg as LA

import phi_policy as phi
import un_policy as un
import heu_bf_policy as bf
import mc_x as MC


def mdp(G, R, RT, L, d, x0, gamma):
    [T, n, A] = R.shape
    T = T+1
    # prelocating

    # unconstrained  policy
#    un_U=np.zeros((n, T))
#    un_Q=np.zeros((T-1, n, A))
#    un_M=np.zeros((T-1, n, n))
#    un_x=np.zeros((n, T))

    # basic feasible policy
    phi_U=np.zeros((n, T))
    phi_Q=np.zeros((T-1, n, A))
    phi_M=np.zeros((T-1, n, n))
    phi_x=np.zeros((n, T))
    phi_opt=np.zeros((1,T-1))

    # heuristic policy-projection
    pro_U=np.zeros((n, T))
    pro_Q=np.zeros((T-1, n, A))
    pro_M=np.zeros((T-1, n, n))
    pro_x=np.zeros((n, T))

    # heuristic policy-backward forward induction
    bf_U=np.zeros((n, T))
    bf_Q=np.zeros((T-1, n, A))
    bf_M=np.zeros((T-1, n, n))
    bf_x=np.zeros((n, T))

    # Initialization
#    un_U[:,[T-1]]=RT
    phi_U[:,[T-1]]=RT
    pro_U[:,[T-1]]=RT
    bf_U[:,[T-1]]=RT


    # Backward Induction
    for j in range(T-2,-1,-1):
        print("Current step of solving phi: ", j)
#        [un_U[:,[j]],un_Q[j,:,:],un_M[j,:,:]]=un.policy(G, R[j,:,:], un_U[:,[j+1]], gamma)
        [phi_U[:,[j]],phi_Q[j,:,:],phi_M[j,:,:],phi_opt[0,j]]=phi.policy(G, R[j,:,:], L, d, phi_U[:,j+1], gamma)
        #[pro_U[:,j],pro_Q[j,:,:],pro_M[j,:,:]]=pro_policy(G, R[j,:,:], d, pro_U[:,j+1], gamma);

#    un_x=MC.mc_x(x0,un_M)
    phi_x=MC.mc_x(x0,phi_M)
    print("Phi_opt", phi_opt)

    TO=10;
    i=0;

    bf_x=cp.deepcopy(phi_x)

    while i <= TO:
        # print(i)
        temp_x=cp.deepcopy(bf_x)

        for j in range(T-2,-1,-1):
            [bf_U[:,[j]],bf_Q[j,:,:],bf_M[j,:,:]]=bf.policy(G, R[j,:,:], L, d, bf_x[:,[j]], bf_U[:,j+1], phi_U[:,j], phi_opt[0,j], gamma)

        bf_x=MC.mc_x(x0,bf_M)

        if LA.norm(bf_x-temp_x,np.inf) < 1e-2:
            break

        i=i+1

    print("phi_u: ")
    print(phi_U)
    print("bf_u: ")
    print(bf_U)
    # print("M shape", np.shape(phi_M))
#    print("phiM: ", phi_M)
    return phi_Q, phi_x, bf_Q, bf_x
#    return un_Q, un_x, phi_Q, phi_x, bf_Q, bf_x








