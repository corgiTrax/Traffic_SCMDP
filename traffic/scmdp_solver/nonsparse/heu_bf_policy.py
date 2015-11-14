# function solves for the heuristic version of one-step safe policy (backward-forward induction)
# input data:
# transition matrix g(A,n,n)
# reward matrix for current epoch rt(n,A)
# density upper bound d(n,1)
# utility/value of states in next step u_next
# discount factor gamma(1,1)
# current density x(n,1)
# reference: opt_ref(1,1)
# reference: utility/value of states in current step u_ref


import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

def policy(g, rt, L, d, x, u_next, u_ref, opt_ref,  gamma):
    # add a small constant to resolve numerical issue
    opt_ref += 10e-5
    [m,temp]=d.shape
    [A, temp, n]=g.shape

    GG=np.zeros((n*n,n*A))

    for i in range(n):
        for j in range(n):
            temp_g=np.zeros((1,A))
            for k in range(A):
                temp_g[:,[k]]=g[k,i,j]

            GG[i*n+j,j*A:j*A+A]=temp_g


    FF=np.zeros((n*n,n*n))

    for i in range(n):
        for j in range(n):
           FF[i*n+j,j*n+i]=1

    RR=np.zeros((n,n*A))

    for i in range(n):
        RR[i,i*A:i*A+A]=rt[i,:]

    LL1=np.zeros((m*n,n*n))
    for i in range(n):
        LL1[i*m:i*m+m,i*n:i*n+n]=L

    LL2=np.zeros((m*n,m*m))
    for i in range(n):
        for j in range(m):
            LL2[i*m:i*m+m,j*m:j*m+m]=L[j,i]*np.eye(m)

    EE=np.zeros((m*n,m))
    for i in range(n):
        EE[i*m:i*m+m,:]= np.eye(m)


    UU=np.zeros((n,n*n))

    for i in range(n):
        UU[i,i*n:i*n+n]=gamma*u_next

    OO=np.zeros((n,n*A))

    for i in range(n):
        OO[i,i*A:i*A+A]=np.ones((1,A))

    DD=np.zeros((m,m*m))
    for i in range(m):
        DD[:,i*m:i*m+m]=d[i]*np.eye(m)

    e_n= np.eye(n)
    e_m=np.eye(m)
    e_mn=np.eye(m*n)
    e_mm= np.eye(m*m)
    e_nA= np.eye(n*A)


    z_1_1= np.zeros((1,1))
    z_1_m= np.zeros((1,m))
    z_1_n= np.zeros((1,n))
    z_1_mm= np.zeros((1,m*m))
    z_1_nn= np.zeros((1,n*n))
    z_1_mn= np.zeros((1,m*n))
    z_1_nA= np.zeros((1,n*A))


    z_m_1= np.zeros((m,1))
    z_m_m= np.zeros((m,m))
    z_m_n= np.zeros((m,n))
    z_m_mm= np.zeros((m,m*m))
    z_m_nn= np.zeros((m,n*n))
    z_m_mn= np.zeros((m,m*n))
    z_m_nA= np.zeros((m,n*A))

    z_n_1= np.zeros((n,1))
    z_n_m= np.zeros((n,m))
    z_n_n= np.zeros((n,n))
    z_n_mm= np.zeros((n,m*m))
    z_n_nn= np.zeros((n,n*n))
    z_n_mn= np.zeros((n,m*n))
    z_n_nA= np.zeros((n,n*A))

    z_mm_1= np.zeros((m*m,1))
    z_mm_m= np.zeros((m*m,m))
    z_mm_n= np.zeros((m*m,n))
    z_mm_mm= np.zeros((m*m,m*m))
    z_mm_nn= np.zeros((m*m,n*n))
    z_mm_mn= np.zeros((m*m,m*n))
    z_mm_nA= np.zeros((m*m,n*A))

    z_nn_1= np.zeros((n*n,1))
    z_nn_m= np.zeros((n*n,m))
    z_nn_n= np.zeros((n*n,n))
    z_nn_mm= np.zeros((n*n,m*m))
    z_nn_nn= np.zeros((n*n,n*n))
    z_nn_mn= np.zeros((n*n,m*n))
    z_nn_nA= np.zeros((n*n,n*A))

    z_mn_1= np.zeros((m*n,1))
    z_mn_m= np.zeros((m*n,m))
    z_mn_n= np.zeros((m*n,n))
    z_mn_mm= np.zeros((m*n,m*m))
    z_mn_nn= np.zeros((m*n,n*n))
    z_mn_mn= np.zeros((m*n,m*n))
    z_mn_nA= np.zeros((m*n,n*A))

    z_nA_1= np.zeros((n*A,1))
    z_nA_m= np.zeros((n*A,m))
    z_nA_n= np.zeros((n*A,n))
    z_nA_mm= np.zeros((n*A,m*m))
    z_nA_nn= np.zeros((n*A,n*n))
    z_nA_mn= np.zeros((n*A,m*n))
    z_nA_nA= np.zeros((n*A,n*A))

    Aeq_r1= np.concatenate((GG,           -FF,          z_nn_mn,     z_nn_mm,     z_nn_n,     z_nn_m,     z_nn_n,      z_nn_m,     z_nn_1), axis=1)
    Aeq_r2= np.concatenate((RR,           z_n_nn,       z_n_mn,      z_n_mm,      z_n_n,      z_n_m,      -e_n,        z_n_m,      z_n_1), axis=1)
    Aeq_r3= np.concatenate((z_n_nA,       UU,           z_n_mn,      z_n_mm,      -e_n,       z_n_m,      e_n,         z_n_m,      z_n_1), axis=1)
    Aeq_r4= np.concatenate((z_mn_nA,      LL1,          e_mn,        -LL2,        z_mn_n,     z_mn_m,     z_mn_n,      EE,         z_mn_1), axis=1)
    Aeq_r5= np.concatenate((OO,           z_n_nn,       z_n_mn,      z_n_mm,      z_n_n,      z_n_m,      z_n_n,       z_n_m,      z_n_1), axis=1)

    Aeq= np.concatenate((Aeq_r1, Aeq_r2, Aeq_r3, Aeq_r4, Aeq_r5), axis=0)
    beq= np.concatenate((z_nn_1,   z_n_1,   z_n_1,   z_mn_1,   np.ones((n,1))), axis=0)

    Aineq_r1= np.concatenate((z_m_nA,     z_m_nn,       z_m_mn,      DD,          z_m_n,      z_m_m,      z_m_n,       -e_m,       z_m_1), axis=1)
    Aineq_r2= np.concatenate((z_n_nA,     z_n_nn,       z_n_mn,      z_n_mm,      z_n_n,      -L.T,       z_n_n,       z_n_m,      np.ones((n,1))), axis=1)
    Aineq_r3= np.concatenate((-e_nA,      z_nA_nn,      z_nA_mn,     z_nA_mm,     z_nA_n,     z_nA_m,     z_nA_n,      z_nA_m,     z_nA_1), axis=1)
    Aineq_r4= np.concatenate((z_mn_nA,    z_mn_nn,      -e_mn,       z_mn_mm,     z_mn_n,     z_mn_m,     z_mn_n,      z_mn_m,     z_mn_1), axis=1)
    Aineq_r5= np.concatenate((z_mm_nA,    z_mm_nn,      z_mm_mn,     -e_mm,       z_mm_n,     z_mm_m,     z_mm_n,      z_mm_m,     z_mm_1), axis=1)
    Aineq_r6= np.concatenate((z_m_nA,     z_m_nn,       z_m_mn,      z_m_mm,      z_m_n,      -e_m,       z_m_n,       z_m_m,      z_m_1), axis=1)
    Aineq_r7=np.concatenate((z_1_nA,      z_1_nn,       z_1_mn,      z_1_mm,      z_1_n,      d.reshape(1,m),      z_1_n,      z_1_m,        -np.ones((1,1))), axis=1)

    c= np.concatenate((z_nA_1,       z_nn_1,      z_mn_1,     z_mm_1,     -x,     z_m_1,        z_n_1,      z_m_1,      -np.zeros((1,1))), axis=0)

    opt=np.zeros([1,1])
    opt[0,0]=opt_ref
    Aineq= np.concatenate((Aineq_r1, Aineq_r2, Aineq_r3, Aineq_r4, Aineq_r5, Aineq_r6, Aineq_r7), axis=0)
    bineq= np.concatenate((d,        u_ref.reshape(n,1),     z_nA_1,     z_mn_1,     z_mm_1,     z_m_1,  opt), axis=0)

    Aeq= matrix(Aeq)
    beq= matrix(beq)
    Aineq= matrix(Aineq)
    bineq= matrix(bineq)
    c= matrix(c)

    sol= solvers.lp(c,Aineq,bineq,Aeq,beq)
    var= np.array(sol['x'])
    # print(var.shape)

    temp_q=var[0:n*A,:]
    Q=temp_q.reshape(n,A)


    # M=np.zeros((n,n))
    # for i in range(n):
    #     M[0:n,:]=var[n*A+i*n:n*A+i*n+n]

    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(A):
                M[i,j]=M[i,j]+g[k,i,j]*Q[j,k]

    U=var[n * A + (n**2 + m**2 + m*n):n*A + (n**2 + m**2 + m*n) +n]
    return U, Q, M






