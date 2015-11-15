# solve for unconstrained deterministic policy
# input data:
# transition matrix g(n,n,A)
# reward matrix for current epoch rt(n,A)
# density upper bound d(n,1)
# utility/value of states in next step u_next
# discount factor gamma(1,1)
import numpy as np

def policy(g, rt, u_next, gamma):
    [A, temp,n]=g.shape
    gg=np.zeros((n,n*A))
    for i in range(A):
        gg[:,i*n:i*n+n]=g[i,:,:]

    qval=np.zeros((n,A))
    Q=np.zeros((n,A))
    V=np.zeros((n,1))

    for i in range(n):
        for a in range(A):
            #temp_g=g[a,:,:]
            qval[i,a]=rt[i,a]+np.dot(gamma*g[a,:,i], u_next)
        val=np.amax(qval[i,:])
        ind=np.argmax(qval[i,:])
        Q[i,ind]=1
        V[i]=val
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(A):
                M[i,j]=M[i,j]+g[k,i,j]*Q[j,k]


    return V, Q,  M
