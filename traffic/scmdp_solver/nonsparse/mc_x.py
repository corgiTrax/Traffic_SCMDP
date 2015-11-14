# density propagation
# input data:
# initial density x0(n,1)
# propagation matrix M(n,n,T-1)

import numpy as np

def mc_x(x0, M):
    [T,temp,n]=M.shape
    # print(T)
    T=T+1
    # print("T",T)

    X=np.zeros((n, T))
    # print(np.shape(X))
    X[:,[0]]=x0

    for i in range(T-1):
        X[:,[i+1]]=np.dot(M[i,:,:], X[:,[i]])

    # print("X shape",np.shape(X))

    return X


